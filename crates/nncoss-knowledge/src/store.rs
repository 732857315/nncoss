use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::index::Index;

const PASSAGES: &str = "passages.jsonl";
const NOTES: &str = "notes.jsonl";
const NOTE_BOOST: f32 = 1.5;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Passage {
    pub id: String,
    pub source: String,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Note {
    pub id: u64,
    pub text: String,
    #[serde(default)]
    pub sources: Vec<String>,
    #[serde(default)]
    pub question: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Passage,
    Note,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    pub kind: Kind,
    pub id: String,
    pub source: String,
    pub text: String,
    pub score: f32,
}

pub struct KnowledgeBase {
    directory: PathBuf,
    passages: Vec<Passage>,
    notes: Vec<Note>,
    entries: Vec<(Kind, usize)>,
    index: Index,
}

fn read_lines<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(path).map_err(|error| format!("无法打开 {}: {error}", path.display()))?;
    let mut items = Vec::new();
    for (number, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|error| format!("读取 {} 失败: {error}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let item = serde_json::from_str(&line)
            .map_err(|error| format!("{} 第 {} 行格式错误: {error}", path.display(), number + 1))?;
        items.push(item);
    }
    Ok(items)
}

fn append_lines<T: Serialize>(path: &Path, items: &[T]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| format!("无法写入 {}: {error}", path.display()))?;
    for item in items {
        let line = serde_json::to_string(item).map_err(|error| error.to_string())?;
        writeln!(file, "{line}")
            .map_err(|error| format!("写入 {} 失败: {error}", path.display()))?;
    }
    file.flush().map_err(|error| error.to_string())
}

fn normalized(text: &str) -> String {
    text.chars().filter(|ch| !ch.is_whitespace()).collect()
}

impl KnowledgeBase {
    pub fn open(directory: &Path) -> Result<Self, String> {
        fs::create_dir_all(directory)
            .map_err(|error| format!("无法创建 {}: {error}", directory.display()))?;
        let mut base = Self {
            directory: directory.to_path_buf(),
            passages: Vec::new(),
            notes: Vec::new(),
            entries: Vec::new(),
            index: Index::new(),
        };
        for passage in read_lines::<Passage>(&directory.join(PASSAGES))? {
            base.insert_passage(passage);
        }
        for note in read_lines::<Note>(&directory.join(NOTES))? {
            base.insert_note(note);
        }
        Ok(base)
    }

    pub fn passages(&self) -> &[Passage] {
        &self.passages
    }

    pub fn notes(&self) -> &[Note] {
        &self.notes
    }

    fn insert_passage(&mut self, passage: Passage) {
        self.index.add(&passage.text);
        self.entries.push((Kind::Passage, self.passages.len()));
        self.passages.push(passage);
    }

    fn insert_note(&mut self, note: Note) {
        self.index.add(&note.text);
        self.entries.push((Kind::Note, self.notes.len()));
        self.notes.push(note);
    }

    pub fn import(&mut self, input: &Path) -> Result<usize, String> {
        let passages = read_lines::<Passage>(input)?;
        if let Some(bad) = passages
            .iter()
            .find(|passage| passage.id.is_empty() || passage.text.trim().is_empty())
        {
            return Err(format!("段落缺少 id 或正文：{}", bad.id));
        }
        append_lines(&self.directory.join(PASSAGES), &passages)?;
        let count = passages.len();
        for passage in passages {
            self.insert_passage(passage);
        }
        Ok(count)
    }

    pub fn search(&self, query: &str, limit: usize) -> Vec<Hit> {
        let mut hits: Vec<Hit> = self
            .index
            .search(query, limit.saturating_mul(4))
            .into_iter()
            .map(|(document, score)| match self.entries[document as usize] {
                (Kind::Passage, position) => {
                    let passage = &self.passages[position];
                    Hit {
                        kind: Kind::Passage,
                        id: passage.id.clone(),
                        source: passage.source.clone(),
                        text: passage.text.clone(),
                        score,
                    }
                }
                (Kind::Note, position) => {
                    let note = &self.notes[position];
                    Hit {
                        kind: Kind::Note,
                        id: format!("note-{}", note.id),
                        source: "笔记".to_string(),
                        text: note.text.clone(),
                        score: score * NOTE_BOOST,
                    }
                }
            })
            .collect();
        hits.sort_by(|left, right| right.score.total_cmp(&left.score));
        hits.truncate(limit);
        hits
    }

    pub fn add_note(
        &mut self,
        text: &str,
        sources: Vec<String>,
        question: Option<String>,
    ) -> Result<Option<u64>, String> {
        let text = text.trim();
        if text.chars().count() < 4 {
            return Ok(None);
        }
        let key = normalized(text);
        if self.notes.iter().any(|note| normalized(&note.text) == key) {
            return Ok(None);
        }
        let id = self.notes.iter().map(|note| note.id).max().unwrap_or(0) + 1;
        let note = Note {
            id,
            text: text.to_string(),
            sources,
            question,
        };
        append_lines(&self.directory.join(NOTES), std::slice::from_ref(&note))?;
        self.insert_note(note);
        Ok(Some(id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn scratch() -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "nncoss-knowledge-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        path
    }

    #[test]
    fn imported_passages_and_notes_survive_reopening() {
        let directory = scratch();
        let input = directory.with_extension("jsonl");
        fs::write(
            &input,
            "{\"id\":\"p1\",\"source\":\"高中/物理\",\"text\":\"牛顿第二定律：F = ma。\"}\n{\"id\":\"p2\",\"source\":\"初中/化学\",\"text\":\"水的化学式是 H2O。\"}\n",
        )
        .unwrap();
        let mut base = KnowledgeBase::open(&directory).unwrap();
        assert_eq!(base.import(&input).unwrap(), 2);
        let id = base
            .add_note(
                "加速度与合力成正比，与质量成反比。",
                vec!["p1".into()],
                Some("牛顿第二定律是什么".into()),
            )
            .unwrap();
        assert_eq!(id, Some(1));
        assert_eq!(
            base.add_note("加速度与合力成正比，与质量成反比。", Vec::new(), None)
                .unwrap(),
            None
        );
        assert_eq!(base.add_note("短", Vec::new(), None).unwrap(), None);
        let reopened = KnowledgeBase::open(&directory).unwrap();
        assert_eq!(reopened.passages().len(), 2);
        assert_eq!(reopened.notes().len(), 1);
        assert_eq!(reopened.notes()[0].sources, ["p1"]);
        let hits = reopened.search("合力 加速度", 2);
        assert_eq!(hits[0].kind, Kind::Note);
        assert_eq!(reopened.search("化学式", 1)[0].id, "p2");
        fs::remove_dir_all(&directory).unwrap();
        fs::remove_file(&input).unwrap();
    }

    #[test]
    fn import_rejects_passages_without_text() {
        let directory = scratch();
        let input = directory.with_extension("jsonl");
        fs::write(&input, "{\"id\":\"p1\",\"source\":\"x\",\"text\":\"  \"}\n").unwrap();
        let mut base = KnowledgeBase::open(&directory).unwrap();
        assert!(base.import(&input).is_err());
        assert!(base.passages().is_empty());
        fs::remove_dir_all(&directory).unwrap();
        fs::remove_file(&input).unwrap();
    }
}
