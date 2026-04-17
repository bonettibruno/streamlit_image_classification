# 🖼️ AI Image Classification Dashboard

Este é um dashboard interativo construído com **Streamlit** que utiliza modelos de Deep Learning (Vision Transformers e ResNet) para classificar imagens.

## 🚀 Funcionalidades
- **Classificação em tempo real**: Upload de imagens e predição instantânea.
- **Analytics**: Gráficos de distribuição de classes e confiança usando Matplotlib.
- **Histórico**: Registro das imagens analisadas durante a sessão.
- **Exportação**: Download dos resultados em CSV.

## 🛠️ Como rodar

Siga os passos abaixo para configurar o ambiente e executar o dashboard na sua máquina:

### 1. Clonar o repositório
```bash
git clone https://github.com/bonettibruno/streamlit_image_classification.git
cd streamlit_image_classification
```

### 2. Criar um ambiente virtual (venv)
É altamente recomendado utilizar um ambiente virtual para isolar as dependências do projeto e evitar conflitos com o seu Python global:

```bash
python -m venv venv
```

### 3. Ativar o ambiente virtual

**No Windows (PowerShell):**
```bash
.\venv\Scripts\activate
```

**No Linux ou macOS:**
```bash
source venv/bin/activate
```

### 4. Instalar as dependências
Com o ambiente virtual ativo, instale os pacotes necessários:

```bash
pip install -r requirements.txt
```

### 5. Executar a aplicação
```bash
streamlit run image_app.py
```