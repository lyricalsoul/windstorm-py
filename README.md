# 🌪️ windstorm
Uma pequena estação meteorológica que utiliza dados históricos do INMET para realizar previsões de tempo utilizando Machine Learning.

> [!IMPORTANT]  
> Este projeto é um POC (Proof of Concept) e não deve ser utilizado para previsões reais.
> O Windstorm é distribuído sem garantias de qualquer tipo e os autores não se responsabilizam por qualquer dano causado pelo uso do software, isentando-se de qualquer responsabilidade decorrente do uso deste programa dentro dos limites da legislação aplicável.

## 📂 estrutura

```
windstorm
├── data_treatment
│   ├──> Scripts para tratamento de dados provenients do BDMEP (Banco de Dados Meteorológicos do INMET)
├── models
│   ├──> Pasta de output do modelo já treinado
├── raw_data
│   ├──> exported_from_inmet
│   │   ├──> Dados puros diretamente do INMET, sem tratamento de dados.
│   ├──> semitreated
│   │   ├──> Dados com algum nível de tratamento - normalmente somente normalização de fields como data.
│   ├──> treated
│   │   ├──> Dados já tratados (em sua maioria). O usado atualmente é MERGED_INMET_SERIES_2013-2024_TREATED_2.csv
├── prediction
│   ├──> Scripts para treinamento dos modelos climatológicos e execução do cliente WS de previsão climática.
```