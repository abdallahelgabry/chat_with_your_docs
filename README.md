# Document Chat Powered by Docling & OCI Generative AI

A Streamlit app that lets you upload a **PDF or Excel file** and have a natural conversation with its contents — powered by Cohere's Command-A model running on Oracle Cloud Infrastructure (OCI) Generative AI.

---

## What It Does

1. **Upload** a PDF or Excel file
2. The app **parses and understands** the document structure using Docling
3. Content is **chunked semantically** and stored in a FAISS vector store
4. You **chat** with your document in natural language — in English or Arabic
5. Every answer is grounded strictly in the document's content

---

## Why Docling?

Most document processing pipelines treat files as raw text dumps. **Docling changes that.**

Developed by IBM Research, Docling is an open-source document intelligence library that understands documents the way humans do — not just as strings of characters, but as structured, meaningful content.

### What makes Docling powerful:

**Deep Document Understanding**
Docling doesn't just extract text. It understands the layout — headings, sections, tables, captions, lists — and preserves that structure throughout the pipeline. This means the AI gets context-aware chunks, not random text fragments.

**Hybrid Chunking**
Instead of splitting text blindly by character count, Docling's `HybridChunker` splits by semantic boundaries. Chunks respect section headings, paragraph flow, and document hierarchy. This directly improves retrieval quality and answer accuracy.

**Broad Format Support**
Docling handles PDF, DOCX, XLSX, PPTX, HTML, images, and URLs — all through a unified `DocumentConverter` interface. No format-specific glue code needed.

**Table & Figure Awareness**
Tables in PDFs are notoriously hard to parse. Docling identifies table boundaries and exports them in a structured way, preserving row/column relationships that standard text extractors destroy.

**Seamless LangChain Integration**
Docling outputs `Document` objects compatible with LangChain's ecosystem, making it a natural fit for RAG (Retrieval-Augmented Generation) pipelines like this one.

---

## Setup

### 1. Install dependencies

```bash
pip install streamlit langchain langchain-community langchain-openai \
            langchain-text-splitters faiss-cpu docling tiktoken \
            oci pillow python-dotenv pdf2image pandas openpyxl
```

### 2. Configure environment variables

Create a `.env` file in the project root

### 3. Run the app

```bash
streamlit run main.py
```

---

##  Supported File Types

| Format | Processing Method |
|---|---|
| PDF | Docling `DocumentConverter` + HybridChunker |
| Excel (`.xlsx`, `.xls`) | Docling with pandas fallback |