# Best Practices: Do's and Don'ts

## Design Patterns

### ✅ DO

- Use `chunk_size=1000` e `chunk_overlap=200` como padrão
- Preserve metadata dos documentos originais
- Inclua sempre citations nas respostas
- Use `k=2-5` documentos para retrieval
- Limpe dados antes de indexar
- Versione seu vector store
- Teste com sample queries

### ❌ DON'T

- Chunking muito pequeno (< 500) ou muito grande (> 2000)
- Perder informações de source (metadata)
- Responder sem citar fontes
- k muito alto (pode confundir LLM)
- Indexar dados ruidosos sem limpeza
- Hardcodar caminhos de arquivos
- Deploy sem testing

---

## Performance

### ✅ DO

- Cache embeddings e queries frequentes
- Processe em batch quando possível
- Use async/await para I/O
- Monitore com LangSmith
- Use reranking para melhor precisão
- Compress vectors para economia
- Monitore latência e throughput

### ❌ DON'T

- Gerar embedding toda vez (cache them)
- Uma operação por vez (use batch)
- Operações síncronas (use async)
- Fazer deploy sem monitoring
- Sempre usar dense search (teste hybrid)
- Desperdiciar memória (comprima vectors)
- Ignorar métricas de performance

---

## Quality

### ✅ DO

- A/B test diferentes abordagens
- Combine métricas automáticas + human eval
- Colete feedback dos usuários continuamente
- Versione modelos e configurations
- Valide retrieval quality regularmente
- Teste edge cases
- Mantenha dataset de evaluation

### ❌ DON'T

- Escolher baseado em intuição (use data)
- Só usar métricas automáticas
- Ignorar feedback de usuários
- Usar sempre mesma versão sem updates
- Assumir que retrieval funciona sem testar
- Testar só casos óbvios
- Esquecer de evaluate mudanças

---

## Production

### ✅ DO

- Health checks em todos os componentes
- Rate limiting em APIs
- Error handling robusto
- Logging estruturado
- Backup regular do vector store
- Environment variables para config
- CI/CD pipeline
- Monitoring de errors

### ❌ DON'T

- Deploy sem health checks
- Permitir unlimited requests
- Deixar errors sem tratamento
- Usar print() para logging
- Indexar só em memória (sem persistência)
- Hardcodar configurações
- Deploy manual sem pipeline
- Ignorar error logs

---

## Security

### ✅ DO

- API keys em environment variables
- Sanitize inputs do usuário
- Rate limiting por user/IP
- Audit logs para compliance
- Data encryption at rest
- HTTPS only
- Access control
- Secret rotation

### ❌ DON'T

- Hardcodar API keys no código
- Usar raw user input diretamente
- Permitir unlimited requests
- Não ter logs de acesso
- Dados em texto plano
- HTTP (use HTTPS)
- Sem controle de acesso
- Mesmas secrets por muito tempo

---

## Code Quality

### ✅ DO

- Type hints em funções
- Docstrings completas
- Comments para lógica complexa
- Unit tests para funções críticas
- Error messages claras
- Configuração external
- Modular design
- Consistent naming

### ❌ DON'T

- Any types everywhere
- Sem documentação
- Assumir que código é óbvio
- Deploy sem tests
- Vague error messages
- Hardcoded values
- Monolithic code
- Inconsistent names

---

## Embeddings

### ✅ DO

- Escolha embeddings adequados ao domínio
- Normalize textos antes de embed
- Teste diferentes models
- Cache embeddings results
- Use appropriate dimensionality
- Considere custo/performance
- Evaluate embedding quality

### ❌ DON'T

- Usar qualquer embedding sem teste
- Indexar textos sem normalizar
- Sempre usar o mesmo model
- Recomputar embeddings
- Usar dimensões muito altas sem necessidade
- Ignorar custo de embeddings
- Não avaliar qualidade

---

## Vector Databases

### ✅ DO

- Choose based on scale requirements
- Test query latency
- Plan for sharding
- Monitor index size
- Regular maintenance
- Backup strategies
- Document migration path
- Consider managed vs self-hosted

### ❌ DON'T

- Choose based on popularity only
- Ignore query performance
- Single point of failure
- Ignore index growth
- Never clean up
- No backup plan
- Lock in without exit strategy
- Assume same config works everywhere

---

## Prompts

### ✅ DO

- Templates consistentes
- Instruction claras
- Examples Few-shot quando útil
- Format de output específico
- Temperature baixo para factual
- Iterative improvement
- A/B test prompts

### ❌ DON'T

- Prompts inconsistentes
- Vague instructions
- Sem examples
- Ambiguous output format
- High temperature para QA
- Set and forget
- Não test optimization

---

## Monitoring

### ✅ DO

- Latency p50, p95, p99
- Error rates
- Throughput (QPS)
- Cost per query
- User satisfaction
- Alerting thresholds
- Dashboard visual
- Log aggregation

### ❌ DON'T

- Só medir média
- Ignorar error rate
- Não medir throughput
- Ignorar custos
- Não perguntar usuários
- Alerts que não disparam
- Métricas em Excel
- Logs descentralizados

---

## Testing

### ✅ DO

- Unit tests
- Integration tests
- End-to-end tests
- Load testing
- Regression tests
- Golden datasets
- Automate tests
- Coverage > 80%

### ❌ DON'T

- Só manual testing
- Sem integration tests
- Deploy sem E2E
- Testar só happy path
- Não testar regressions
- Sem golden reference
- Manual testing only
- Low coverage

---

## Documentation

### ✅ DO

- README completo
- API documentation
- Architecture diagrams
- Runbooks
- Troubleshooting guide
- Changelog
- Code comments
- Examples

### ❌ DON'T

- README vago
- Sem docs da API
- Diagramas desatualizados
- Sem runbooks
- Sem troubleshooting
- Sem versionamento
- Código sem comments
- Examples que não funcionam

---

## Scalability

### ✅ DO

- Horizontal scaling
- Load balancing
- Caching layers
- Async processing
- Queue systems
- Auto-scaling
- Database sharding
- CDN para assets

### ❌ DON'T

- Só vertical scaling
- Single point of failure
- Sem caching
- Synchronous only
- Sem queue
- Manual scaling
- Single database
- Assets sem CDN

---

## Error Handling

### ✅ DO

- Graceful degradation
- Retry logic
- Circuit breakers
- Fallback responses
- Clear error messages
- Log errors
- Alert on critical
- Recovery procedures

### ❌ DON'T

- Fail hard
- Infinite retry
- Cascade failures
- Generic errors
- Vague messages
- Ignorar errors
- Dormir em errors críticos
- Sem recovery plan

---

## Cost Optimization

### ✅ DO

- Cache results
- Batch operations
- Right-size instances
- Spot instances
- Reserved capacity
- Monitor usage
- Optimize embeddings
- Compress data

### ❌ DON'T

- Compute same thing twice
- Process one by one
- Over-provision
- On-demand only
- No commitment
- Ignorar billing
- Wasteful embeddings
- Duplicar dados

---

## Checklist Final

Antes de production, verifique:

- [ ] Design patterns seguidos
- [ ] Performance testado
- [ ] Quality metrics definidas
- [ ] Security audit
- [ ] Code review completo
- [ ] Testing coverage
- [ ] Documentation atualizada
- [ ] Monitoring ativo
- [ ] Scalability planejada
- [ ] Error handling robusto
- [ ] Cost optimization
- [ ] Backup e recovery
