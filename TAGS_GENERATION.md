# Tag Generation Guidelines

This document provides guidelines for automatically tagging test cases based on their extraction requirements. Each test case can be analyzed against these tag definitions to determine which extraction capabilities are needed.

## Available Tags

### STRUCTURAL_EXTRACTION
For extracting data from structured HTML elements with clear hierarchies and patterns like:
- Tables
- Navigation menus 
- Directory listings
- Ordered/unordered lists

### METADATA_EXTRACTION
For extracting configuration and metadata like:
- Meta tags
- Site settings
- Social media metadata
- SEO data

### MEDIA_EXTRACTION
For extracting media assets and their attributes:
- Image URLs and metadata
- Video embeds
- Audio files
- Base64 encoded media

### SEMANTIC_EXTRACTION
For extracting meaning from content:
- Article text
- Comment threads
- User profiles
- Natural language content

### ANALYTICS_EXTRACTION
For extracting numerical and statistical data:
- Page view counts
- User metrics
- Engagement statistics
- Numerical aggregations

## Prompt

```
Test cases should be analyzed to determine which extraction capabilities they require. Multiple tags can be applied if a test case spans multiple extraction types. The tags help categorize test cases and ensure appropriate extraction methods are used. Add the tags property to the test case and identify which tags are appropriate for this task.

STRUCTURAL_EXTRACTION


For tasks involving extracting data from structured elements like tables, lists, and hierarchical navigation
Example tasks: Menu structures, pricing tables, directory listings
Usually has clear parent-child relationships and consistent patterns


METADATA_EXTRACTION


For tasks extracting page/site metadata and configuration information
Example tasks: Meta tags, site settings, social media metadata, SEO data
Often found in document head or configuration sections


MEDIA_EXTRACTION


For tasks involving images, videos, audio and their associated attributes
Example tasks: Image URLs, video embeds, image metadata, base64 encoded media
Includes both media locations and associated metadata


SEMANTIC_EXTRACTION


For tasks requiring understanding of content meaning and relationships
Example tasks: Article content, comment threads, user profiles
Requires understanding context and content relationships


ANALYTICS_EXTRACTION


For tasks involving numerical data, statistics, and metrics
Example tasks: Page views, user counts, engagement metrics
Often involves parsing and aggregating numerical data
```