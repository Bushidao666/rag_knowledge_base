# Best Practices: Chunking Strategies

## Design Patterns

### ✅ DO

- Use RecursiveCharacterTextSplitter como padrão
- Start com chunk_size=1000, chunk_overlap=200
- Teste com sample data antes de production
- Preserve document metadata sempre
- Ajust chunk_size baseado no use case
- Use semantic separators quando possível
- Monitor chunk statistics

### ❌ DON'T

- Usar CharacterTextSplitter para tudo
- Hardcode chunk_size sem testar
- Processar sem validation
- Perder information de source
- Usar mesmo chunk_size para todos os casos
- Ignorar document structure
- Skip quality checks

---

## Chunk Size

### ✅ DO

- Ajust baseado no conteúdo
- Teste diferentes sizes (500, 1000, 1500)
- Consider context window do LLM
- Balance precision vs. context
- Smaller (500) para technical docs
- Larger (1500-2000) para narrativa

### ❌ DON'T

- Usar muito pequeno (<500) sempre
- Usar muito grande (>2000) sempre
- Ignorar tipo de documento
- Não considerar cost
- Set and forget

---

## Chunk Overlap

### ✅ DO

- 20-30% do chunk_size
- Test diferentes overlaps
- Consider content type
- Higher (30%) para conversational
- Lower (20%) para factual

### ❌ DON'T

- Zero overlap (context loss)
- Too high (50%+) redundancy
- Same overlap for all cases
- Ignore redundancy cost
- Não calcular overlap effectiveness

---

## Separators

### ✅ DO

- Order by semantic importance
- Use natural boundaries
- Test separator combinations
- Add start_index para tracking
- Custom separators para código
- Preserve document structure

### ❌ DON'T

- Random separator order
- Break at arbitrary points
- Ignore content type
- No testing de separators
- Forget document hierarchy
- Over-engineer separators

---

## Content Type

### ✅ DO

- Technical docs: smaller chunks (500-800)
- Narrative docs: larger chunks (1000-1500)
- Code: even smaller (300-500)
- Conversational: medium (800-1000)
- Tables: preserve structure
- Headers: keep with content

### ❌ DON'T

- Same strategy para all types
- Break code functions
- Separate headers from content
- Ignore table structure
- Not preserve context
- One-size-fits-all

---

## Metadata

### ✅ DO

- Always preserve metadata
- Add chunk_id
- Track position (start_index)
- Include document source
- Version documents
- Validate metadata completeness

### ❌ DON'T

- Lose source information
- No chunk tracking
- Ignore document version
- Overwrite metadata
- Inconsistent keys
- Skip validation

---

## Performance

### ✅ DO

- Batch processing para large datasets
- Lazy loading quando possível
- Profile splitting time
- Cache processed chunks
- Parallel processing
- Monitor memory usage
- Use streaming para big files

### ❌ DON'T

- Load all in memory
- Process sequentially only
- No performance testing
- No caching
- Single-threaded only
- Ignore memory limits
- Block on large files

---

## Quality Assurance

### ✅ DO

- Validate após cada split
- Check chunk distribution
- Verify overlap effectiveness
- Sample review manual
- Monitor retrieval quality
- A/B test different params
- Track metrics over time

### ❌ DON'T

- Trust automático splitting
- Ignore distribution skew
- No manual review
- Skip retrieval testing
- No A/B testing
- Metrics drift
- No continuous monitoring

---

## Custom Splitters

### ✅ DO

- Extend TextSplitter class
- Test with edge cases
- Document your logic
- Benchmark performance
- Handle empty inputs
- Raise clear errors
- Provide examples

### ❌ DON'T

- Reinvent the wheel
- No edge case testing
- Undocumented logic
- Poor performance
- Crashes on empty
- Silent failures
- No usage examples

---

## Large Documents

### ✅ DO

- Stream processing
- Checkpoint progress
- Handle interruptions
- Validate incremental
- Use async I/O
- Memory management
- Resume capability

### ❌ DON'T

- Load entire document
- No progress tracking
- All-or-nothing processing
- No checkpointing
- Blocking I/O
- Memory leaks
- No recovery

---

## Structured Documents

### ✅ DO

- Preserve document hierarchy
- Keep headers with content
- Handle nested structures
- Tables as units
- Semantic boundaries first
- Recursive splitting
- Cross-references

### ❌ DON'T

- Ignore document structure
- Separate headers
- Break hierarchical data
- Mix unrelated content
- Arbitrary splits
- Flat structure
- Lost relationships

---

## Production

### ✅ DO

- Health checks
- Configuration external
- Version chunks
- Rollback capability
- Monitoring
- Alerting
- Documentation

### ❌ DON'T

- No health checks
- Hardcoded config
- No versioning
- No rollback
- No monitoring
- Silent failures
- Poor docs

---

## Testing

### ✅ DO

- Unit tests
- Integration tests
- Property-based tests
- Edge case coverage
- Performance benchmarks
- Regression tests
- Golden data

### ❌ DON'T

- Only happy path
- No integration tests
- No edge cases
- No performance testing
- No regression suite
- Flaky tests
- No reference data

---

## Monitoring

### ✅ DO

- Track chunk distribution
- Monitor overlap effectiveness
- Count average chunk size
- Track processing time
- Error rate monitoring
- Quality metrics
- Dashboard visualization

### ❌ DON'T

- No distribution tracking
- Ignore overlap metrics
- No size monitoring
- No performance tracking
- No error monitoring
- Terminal only
- No visualization

---

## Security

### ✅ DO

- Validate input size
- Sanitize custom logic
- Handle sensitive data
- Access control
- Encryption at rest
- Audit logging
- Data retention policy

### ❌ DON'T

- Unrestricted input
- Unsafe custom code
- Expose sensitive data
- No access control
- Plain text storage
- No audit trail
- Indefinite retention

---

## Documentation

### ✅ DO

- Document strategy
- Explain parameters
- Provide examples
- Record changes
- Best practices guide
- Troubleshooting section
- API documentation

### ❌ DON'T

- Vague documentation
- No parameter explanation
- No examples
- No change log
- No guidance
- No troubleshooting
- No API docs

---

## Checklist Final

Before production:
- [ ] Default strategy tested
- [ ] Parameters validated
- [ ] Quality metrics defined
- [ ] Performance benchmarked
- [ ] Monitoring active
- [ ] Tests written
- [ ] Documentation complete
- [ ] Security reviewed
