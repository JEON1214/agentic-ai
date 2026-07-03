## Final Critical Reflection

### 1. Which chunking strategy returned the most relevant text for your query? Did it capture the entire sentence context or was it cut off?

After running my program, the **Fixed-Size chunking strategy** returned the most relevant text for my query, "Why does overlap help when chunking a document?" The retrieved chunk contained the important information about overlap preserving meaning and context. However, I noticed that the returned text was **cut off before the sentence was completed**, ending with "Overlap helps preserve meaning when a sent". This happened because fixed-size chunking divides the document based on a fixed number of characters instead of complete sentences or paragraphs. Although the important keywords were still present, the incomplete sentence reduced the readability and could make the retrieved information less clear.

### 2. What happened to the text structure in Fixed-Size Chunk #2 vs. Paragraph Chunk #2? Identify how boundaries changed word availability.

The Fixed-Size Chunk #2 split the document according to the specified character limit, which caused part of a sentence to be separated from the rest of its content. As a result, some words and ideas were incomplete or appeared in different chunks. In contrast, the Paragraph Chunking strategy preserved the original paragraph boundaries, keeping related sentences together. This maintained the complete meaning of the text and provided better context because the words and ideas were not interrupted by arbitrary character limits.

### 3. Hypothetical Application: Imagine you are building a production AI system for a company's internal HR manual handbook. Why might relying exclusively on Fixed-Size character chunking create bad answers for employees?

If I were developing an AI system for a company's HR handbook, relying only on Fixed-Size character chunking could lead to incorrect or incomplete answers. Important policies, procedures, or requirements might be split across different chunks, causing the AI to retrieve only part of the information. Employees could receive incomplete instructions or misunderstand company policies because the retrieved text lacks the full context. Using structural chunking methods, such as paragraph-based chunking, would help preserve complete ideas and provide more accurate responses.

### 4. The Metadata Payload: Why do we spend computing effort storing things like chunk_index and strategy inside the database alongside raw vectors? Why can't we just store the text string alone?

Metadata is important because it provides additional information about each stored chunk beyond the text itself. The `chunk_index` allows us to identify the chunk's original position within the document, while the `strategy` tells us which chunking method produced it. This information is valuable for debugging, evaluating retrieval performance, and comparing different chunking strategies. If we stored only the text, we would not know where it came from or how it was created, makin