use std::collections::{HashMap, HashSet};

const K1: f32 = 1.2;
const B: f32 = 0.75;

fn is_cjk(ch: char) -> bool {
    matches!(ch, '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}' | '\u{f900}'..='\u{faff}')
}

fn flush_run(run: &mut Vec<char>, output: &mut Vec<String>) {
    match run.len() {
        0 => {}
        1 => output.push(run[0].to_string()),
        _ => output.extend(run.windows(2).map(|pair| pair.iter().collect::<String>())),
    }
    run.clear();
}

fn flush_word(word: &mut String, output: &mut Vec<String>) {
    if !word.is_empty() {
        output.push(std::mem::take(word));
    }
}

pub fn terms(text: &str) -> Vec<String> {
    let mut output = Vec::new();
    let mut run = Vec::new();
    let mut word = String::new();
    for ch in text.chars().flat_map(char::to_lowercase) {
        if is_cjk(ch) {
            flush_word(&mut word, &mut output);
            run.push(ch);
        } else if ch.is_ascii_alphanumeric() {
            flush_run(&mut run, &mut output);
            word.push(ch);
        } else {
            flush_word(&mut word, &mut output);
            flush_run(&mut run, &mut output);
        }
    }
    flush_word(&mut word, &mut output);
    flush_run(&mut run, &mut output);
    output
}

#[derive(Default)]
pub struct Index {
    vocabulary: HashMap<String, u32>,
    postings: Vec<Vec<(u32, u32)>>,
    lengths: Vec<u32>,
    total: u64,
}

impl Index {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.lengths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lengths.is_empty()
    }

    pub fn add(&mut self, text: &str) -> u32 {
        let document = self.lengths.len() as u32;
        let list = terms(text);
        let mut counts: HashMap<u32, u32> = HashMap::new();
        for term in &list {
            let next = self.vocabulary.len() as u32;
            let id = *self.vocabulary.entry(term.clone()).or_insert(next);
            if id as usize == self.postings.len() {
                self.postings.push(Vec::new());
            }
            *counts.entry(id).or_default() += 1;
        }
        for (term, count) in counts {
            self.postings[term as usize].push((document, count));
        }
        self.lengths.push(list.len() as u32);
        self.total += list.len() as u64;
        document
    }

    pub fn search(&self, query: &str, limit: usize) -> Vec<(u32, f32)> {
        if self.lengths.is_empty() || limit == 0 {
            return Vec::new();
        }
        let count = self.lengths.len() as f32;
        let average = (self.total as f32 / count).max(1.0);
        let mut scores: HashMap<u32, f32> = HashMap::new();
        let mut seen = HashSet::new();
        for term in terms(query) {
            if !seen.insert(term.clone()) {
                continue;
            }
            let Some(&id) = self.vocabulary.get(&term) else {
                continue;
            };
            let list = &self.postings[id as usize];
            let frequency = list.len() as f32;
            let weight = ((count - frequency + 0.5) / (frequency + 0.5) + 1.0).ln();
            for &(document, hits) in list {
                let hits = hits as f32;
                let length = self.lengths[document as usize] as f32;
                let score =
                    weight * hits * (K1 + 1.0) / (hits + K1 * (1.0 - B + B * length / average));
                *scores.entry(document).or_default() += score;
            }
        }
        let mut ranked: Vec<(u32, f32)> = scores.into_iter().collect();
        ranked.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
        ranked.truncate(limit);
        ranked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terms_split_chinese_into_bigrams_and_keep_ascii_words() {
        assert_eq!(
            terms("牛顿第二定律 F=ma"),
            ["牛顿", "顿第", "第二", "二定", "定律", "f", "ma"]
        );
        assert_eq!(terms("水"), ["水"]);
        assert_eq!(terms("Python3 的 list"), ["python3", "的", "list"]);
        assert!(terms("，。!?").is_empty());
    }

    #[test]
    fn search_ranks_the_matching_document_first() {
        let mut index = Index::new();
        index.add("光合作用是绿色植物利用光能把二氧化碳和水转化为有机物的过程。");
        index.add("牛顿第二定律指出物体的加速度与所受合力成正比。");
        index.add("二分查找每次把搜索区间缩小一半。");
        let hits = index.search("什么是牛顿第二定律", 3);
        assert_eq!(hits[0].0, 1);
        assert!(index.search("量子纠缠", 3).is_empty());
        assert_eq!(index.search("二分查找", 1).len(), 1);
    }

    #[test]
    fn empty_index_returns_nothing() {
        let index = Index::new();
        assert!(index.is_empty());
        assert!(index.search("任何问题", 5).is_empty());
    }
}
