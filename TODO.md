# TODO / Future Features

## 🔮 Planned Features

### PDF to Markdown Conversion with Structure Preservation
- **Priority**: Medium
- **Complexity**: High
- **Description**: Convert PDF documents to Markdown format while preserving structural elements (headers, lists, tables)
- **Approach Options**:
  1. Font size/weight heuristics to detect headers
  2. Layout analysis using PDF font metadata
  3. Integration with external tools (Marker, Docling)
- **API**: `extractTextFromPdfAsMarkdown(fileBytes)`
- **Benefits**: Enable structure-aware chunking (`markdown_chunk()`) for PDF documents

### Korean NLP Package (`mobile_rag_engine_ko`)
- **Priority**: Low (on user demand)
- **Complexity**: Medium
- **Description**: Separate package with lindera Korean morpheme analyzer for advanced BM25 tokenization
- **Features**:
  - Korean morpheme analysis (형태소 분석)
  - Better partial matching ("삼성" → "삼성전자")
  - Reduced spacing error sensitivity
- **Trade-offs**: +25MB binary size (Korean dictionary)
- **API**: Same as `mobile_rag_engine`, drop-in replacement

### Prompt Engineering & Robustness
- **Priority**: Low
- **Complexity**: Low
- **Description**: Improvements for metadata handling in LLM prompts
- **Features**:
  - **Metadata Token Budget**: Include metadata length in `ContextBuilder`'s token calculation to prevent truncation
  - **Safety**: Wrap XML content in `CDATA` or escape special characters to handle complex metadata values safely

---

## ✅ Completed Features
- [x] PDF text extraction with page number removal
- [x] DOCX text extraction
- [x] Smart dehyphenation (cross-page word joining)
- [x] Markdown structure-aware chunking
- [x] Code block preservation with language detection
- [x] Table preservation in Markdown
- [x] Header path inheritance for context
