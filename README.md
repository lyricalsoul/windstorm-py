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
│   ├──> Modelos de Machine Learning para previsão de tempo
├── hw_control
│   ├──> Controle de hardware (Arduino/Firmata) para coleta contínua de dados
├── training
│   ├──> Scripts para treinamento dos modelos climatológicos
```