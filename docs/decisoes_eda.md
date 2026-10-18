# FIAP Tech Challenge Fase 2 — Registro das decisões de EDA e pré-processamento

**Fonte de dados:** `data/raw/WineQT.csv` (1.143 amostras, 11 características físico-químicas, `quality` e `Id`). **Sem incorporação de dados externos.**

| Tema | Constatação | Decisão e justificativa |
|---|---|---|
| Variável-alvo | Notas originais de 3 a 8 | Criar classe 1 para `quality >= 7`, classe 0 para `< 7`, seguindo o enunciado |
| `Id` | 1.143 IDs únicos | Usar somente como identificador/rastreabilidade. Excluir das preditoras |
| `quality` original | Define diretamente a classe binária | Excluir das preditoras para evitar vazamento direto do alvo |
| Dados ausentes | Nenhum nas 13 colunas | Não imputar valores inexistentes |
| Duplicações | 125 linhas extras; 114 grupos repetidos; nenhuma classe conflitante nesses perfis | Manter registros, por não ser possível saber se são amostras distintas; agrupar perfis idênticos na divisão e na CV. Limitação: repetição atribui peso a determinados perfis |
| Outliers | IQR de 1,5 detecta 295 amostras com pelo menos uma observação atípica | Manter neste experimento; outlier estatístico não é sinônimo de erro. Qualquer tratamento posterior deve ser justificado e validado no treino |
| Consistência | Nenhuma violação nos seis testes de plausibilidade definidos | Preservar observações; controles não provam ausência de qualquer erro de medição |
| Classes desbalanceadas | Classe 0: 984 (86,09%); classe 1: 159 (13,91%) | Preservar e monitorar proporções na divisão. Avaliar métricas apropriadas para classe 1 e possíveis ajustes de peso na modelagem |
| Separação | 914 treino, 229 teste; nenhum grupo idêntico em ambos; estratificação próxima da original | Versionar `data/processed/divisao_treino_teste.csv` como referência do grupo; reservar o teste para avaliação final |
| Padronização | 11 variáveis com diferentes ordens de grandeza | `StandardScaler` ajustado apenas no treino e aplicado ao teste. Em CV, colocá-lo dentro de um pipeline ajustado por dobra |
| Feature engineering | Proporção média de SO₂ livre maior na classe 1; correlação fraca no treino (r ≈ 0,1325) | `free_so2_ratio = free sulfur dioxide / total sulfur dioxide` é **opcional e experimental**. Comparar por CV no treino antes de escolher |
| Correlações | `alcohol` (+0,404), `volatile acidity` (-0,305), `citric acid` (+0,251), `sulphates` (+0,208) com classe alta | Registrar como associações exploratórias, não provas causais ou seleção automática de preditores |

## Limitações e próximos passos

1. As análises descrevem apenas a amostra disponibilizada; não demonstram relações causais.
2. A manutenção de perfis repetidos pode pesar observações repetidas; grupos idênticos são mantidos juntos nas divisões para reduzir vazamento.
3. Não existe comprovação ainda de ganho preditivo da característica criada; validação do grupo deverá determinar sua utilidade.
4. Modelos devem ser comparados utilizando a mesma divisão e métricas; conjunto de teste é reservado para avaliação final.

Os resultados são documentados em `notebooks/01_eda.ipynb` e `notebooks/02_preprocessamento.ipynb`.
