# ML Pipeline - GitHub Actions

Este projeto implementa um pipeline automatizado de Machine Learning que executa a cada push no repositório.

## Estrutura do Projeto

```
ml-pipeline/
├── data/
│   └── sample.csv          # Dataset de exemplo
├── .github/
│   └── workflows/
│       └── ml.yml          # Workflow do GitHub Actions
├── train.py                # Script de treinamento
├── requirements.txt        # Dependências Python
└── README.md              # Este arquivo
```

## Funcionalidades

- **Treinamento Automatizado**: Executa a cada push no repositório
- **Métricas Completas**: Calcula acurácia, precisão, recall e F1-score
- **Relatório Detalhado**: Gera `report.txt` com todas as métricas
- **Artefatos**: O relatório é disponibilizado como artefato no GitHub

## Como Usar

1. Faça push das alterações no repositório
2. O workflow será executado automaticamente
3. Acesse a aba "Actions" no GitHub para ver o progresso
4. Baixe o artefato "classification-report" para obter o relatório

## Métricas Incluídas

- **Acurácia**: Proporção de predições corretas
- **Precisão**: Proporção de predições positivas que são corretas
- **Recall**: Proporção de casos positivos identificados corretamente
- **F1-Score**: Média harmônica entre precisão e recall

## Dependências

- pandas >= 1.5.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
