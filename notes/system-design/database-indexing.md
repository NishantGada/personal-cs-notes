# Database Indexing

there are multiple types of database indexing techniques/algorithms that we can use depending on our specific requirements and use-cases. 

- B-Trees
- Hash Index
- Geospacial Indexing
- Inverted Index

### B-Trees
  - the most commonly used indexing technique because of near O(1) retrieval speed (can be compared to a Hashmap)
  - allows queries by range and sorting (hashmap doesn't allow these)
  - inefficient when searching/querying for entire strings in your DB (unless you're searching for strings with a known prefix)


### Hash Index
  - as the name suggests - uses a hash map to store your data in the form of key-value pairs
  - useful only when you know exactly what you're looking for
  - doesn't support query/search by sorting or within a range


### Geospatial Indexing
  - as the name suggest, super useful when querying data based on latitude/longitude/geographical data
  - common use-cases are finding nearby restaurants/ finding friends in your city/ within a radius/ etc
  - there are multiple ways to do this:
      - Geo-hashing
      - Quad Trees
      - R-Trees


### Inverted Index
  - specifically useful when searching/querying by a string
  - core concept is that we map each unique word, that occurs in our data, to its respective page/location of occurrence
  - something like:
      - "pizza" -> [doc1, doc2]
      - "college" -> [doc2, doc4]
  - we can then directly access the specific pages/docs by this mapping

### REFERENCE
  - https://youtu.be/BHCSL_ZifI0?si=XULlC65X4qQC2gff