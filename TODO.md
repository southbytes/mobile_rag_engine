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

---

## ✅ Completed Features
- [x] PDF text extraction with page number removal
- [x] DOCX text extraction
- [x] Smart dehyphenation (cross-page word joining)
- [x] Markdown structure-aware chunking
- [x] Code block preservation with language detection
- [x] Table preservation in Markdown
- [x] Header path inheritance for context
