# RAG Context Comprehensive: 프롬프트 압축 구현

> **버전**: v0.4.0  
> **최종 업데이트**: 2025-12-23  
> **상태**: Phase 1 완료

---

## 개요

REFRAG(REpresentation For RAG) 논문의 핵심 개념을 mobile_rag_engine에 경량화하여 적용한 프롬프트 압축 기능입니다. 모바일 온디바이스 LLM의 제한된 컨텍스트 윈도우를 효율적으로 활용합니다.

### 핵심 아이디어
```
검색된 문맥 전체를 LLM에 넣지 말고, 
정보 밀도가 높은 부분만 선별하여 전달하라
```

---

## 구현 현황

### Phase 1: 규칙 기반 압축 ✅ 완료

| 기능 | 구현 위치 | 설명 |
|-----|----------|-----|
| 문장 분리 | `compression_utils.rs` | 한국어/영어 Unicode 기반 |
| 불용어 필터링 | `compression_utils.rs` | 한국어 조사/영어 관사 등 |
| 중복 문장 제거 | `compression_utils.rs` | FNV-1a 해시 기반 |
| 압축 파이프라인 | `compression_utils.rs` | 위 기능 통합 |
| Dart 서비스 | `prompt_compressor.dart` | Flutter API 제공 |
| ContextBuilder 통합 | `context_builder.dart` | `buildWithCompression()` |
| UI 컨트롤 | `rag_chat_screen.dart` | 압축 레벨 선택 메뉴 |

### Phase 2: 유사도 기반 문장 선택 ✅ 완료

| 기능 | 상태 | 설명 |
|-----|------|-----|
| 문장별 임베딩 생성 | ✅ 완료 | 기존 EmbeddingService 활용 |
| 쿼리-문장 유사도 계산 | ✅ 완료 | 코사인 유사도 |
| 상위 K개 문장 선택 | ✅ 완료 | `scoreSentences()` 메서드 |

### Phase 3: 청크 임베딩 사전 계산 📋 계획

| 기능 | 상태 | 설명 |
|-----|------|-----|
| 인덱싱 시 압축 임베딩 생성 | 미구현 | 추론 시 연산 절약 |
| DB 스키마 확장 | 미구현 | `compressed_embedding` 컬럼 |

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    Flutter Layer                         │
├─────────────────────────────────────────────────────────┤
│  PromptCompressor.compress()                             │
│       ↓                                                  │
│  ContextBuilder.buildWithCompression()                   │
│       │                                                  │
└───────┼─────────────────────────────────────────────────┘
        │ FFI (flutter_rust_bridge)
        ↓
┌─────────────────────────────────────────────────────────┐
│                     Rust Layer                           │
├─────────────────────────────────────────────────────────┤
│  compression_utils.rs                                    │
│  ├── split_sentences()      문장 분리                    │
│  ├── filter_stopwords()     불용어 제거                  │
│  ├── sentence_hash()        중복 탐지                    │
│  └── compress_text()        통합 파이프라인              │
└─────────────────────────────────────────────────────────┘
```

---

## API 사용법

### 1. PromptCompressor 직접 사용
```dart
import 'package:mobile_rag_engine/mobile_rag_engine.dart';

final compressed = await PromptCompressor.compress(
  chunks: searchResults,
  level: CompressionLevel.balanced,  // minimal, balanced, aggressive
  maxTokens: 2000,
  language: 'ko',  // 'ko' or 'en'
);

print('압축률: ${(compressed.ratio * 100).toStringAsFixed(1)}%');
print('절약 토큰: ~${compressed.estimatedTokensSaved}');
```

### 2. ContextBuilder 통합
```dart
final context = await ContextBuilder.buildWithCompression(
  searchResults: chunks,
  tokenBudget: 2000,
  compressionLevel: 1,  // 0=minimal, 1=balanced, 2=aggressive
  language: 'ko',
  strategy: ContextStrategy.relevanceFirst,
);
```

### 3. Rust 직접 호출
```dart
import 'package:mobile_rag_engine/src/rust/api/compression_utils.dart';

final result = await compressText(
  text: longText,
  maxChars: 8000,
  options: CompressionOptions(
    removeStopwords: true,
    removeDuplicates: true,
    language: 'ko',
    level: 1,
  ),
);
```

### 4. Phase 2: 유사도 기반 압축
```dart
// 1. 먼저 앱 초기화 시 EmbeddingService 등록
PromptCompressor.setEmbeddingService(EmbeddingService.embed);

// 2. 쿼리 임베딩 생성
final queryEmbedding = await EmbeddingService.embed(query);

// 3. 유사도 기반 압축
final compressed = await PromptCompressor.compressWithSimilarity(
  chunks: searchResults,
  queryEmbedding: queryEmbedding,
  maxSentences: 15,
  minSimilarity: 0.2,
  level: CompressionLevel.balanced,
);
```

---

## 압축 레벨

| 레벨 | 값 | 설명 | 예상 압축률 |
|-----|---|------|-----------|
| Minimal | 0 | 중복 문장 제거만 | 5~15% |
| Balanced | 1 | + 경량 필터링 | 15~30% |
| Aggressive | 2 | + 불용어 제거 | 25~40% |

---

## 불용어 목록

### 한국어 (기본 내장)
```
조사: 은, 는, 이, 가, 을, 를, 의, 에, 에서, 으로, 로, 와, 과, ...
접속: 그리고, 그러나, 그래서, 하지만, 또한, ...
대명사: 이것, 그것, 저것, 여기, 거기, ...
```

### 영어 (기본 내장)
```
관사: the, a, an
전치사: to, of, in, for, on, with, ...
대명사: i, you, he, she, it, they, ...
```

---

## 테스트 현황

### Rust 테스트 (9개 통과)
```bash
cd rust && cargo test compression_utils
```
- `test_split_sentences_korean` ✅
- `test_split_sentences_english` ✅
- `test_sentence_hash_identical` ✅
- `test_sentence_hash_different` ✅
- `test_filter_stopwords_english` ✅
- `test_compress_text_removes_duplicates` ✅
- `test_compress_text_respects_max_chars` ✅
- `test_compress_text_simple` ✅
- `test_should_compress` ✅

### Dart 테스트 (5개 통과)
```bash
flutter test test/prompt_compressor_test.dart
```

---

## 추후 개발 사항

1. **Phase 2**: 유사도 기반 문장 선택 (REFRAG Sense 단계 대체)
2. **Phase 3**: 청크 임베딩 사전 계산 (REFRAG Compress 고도화)
3. **도메인별 불용어**: 법률, 의료 분야 커스터마이징

---

## 관련 파일

| 파일 | 역할 |
|-----|-----|
| [compression_utils.rs](file:///Users/dev_bh/Desktop/toys/pub_package/mobile_rag_engine/rust/src/api/compression_utils.rs) | Rust 압축 유틸리티 |
| [prompt_compressor.dart](file:///Users/dev_bh/Desktop/toys/pub_package/mobile_rag_engine/lib/services/prompt_compressor.dart) | Dart 서비스 |
| [context_builder.dart](file:///Users/dev_bh/Desktop/toys/pub_package/mobile_rag_engine/lib/services/context_builder.dart) | 컨텍스트 빌더 (확장) |
| [prompt_compressor_test.dart](file:///Users/dev_bh/Desktop/toys/pub_package/mobile_rag_engine/test/prompt_compressor_test.dart) | 유닛 테스트 |

---

## 참고 자료

- [REFRAG 논문 요약](file:///Users/dev_bh/Desktop/toys/pub_package/mobile_rag_engine/refrag-develop.md)
- [하이브리드 RAG 아키텍처 가이드](file:///Users/dev_bh/Desktop/toys/pub_package/mobile_rag_engine/docs/guides/hybrid_rag_architecture_guide.md)
