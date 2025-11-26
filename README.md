# Processamento Gráfico: Fundamentos - Grau B, 2025/2, Unisinos
> Este programa foi desenvolvido como parte da disciplina *Processamento Gráfico* com foco na exploração e aplicação de técnicas de processamento de imagem. O projeto teve como objetivo desenvolver um protótipo de um aplicativo de edição de imagens e vídeo (inspirado nos stories do Instagram).


## Editor de Imagens em Python (OpenCV + Dear PyGui)

- **Integrantes:**  
  - Cássio F. Braga  
  - Gabriel C. Walber  
  - Patrícia Nagel  

- **Professora:** Rossana Baptista Queiroz  

## Sobre o projeto

Este repositório contém um **editor de imagens simples** desenvolvido em Python, utilizando **OpenCV** para processamento de imagens e **Dear PyGui** para interface gráfica. 
O software permite carregar imagens ou usar a câmera em tempo real, aplicar efeitos visuais e salvar o resultado.

## Funcionalidades

### Modos de uso
- **Modo Imagem** – Trabalha com uma imagem carregada do sistema.
- **Modo Câmera** – Processamento em tempo real usando webcam.

### Efeitos disponíveis
- Blur Gaussiano  
- Escala de Cinza  
- Detecção de Bordas (Canny)  
- Seleção de Canal (R, G, B)  
- Sharpen (nitidez)
- Inverter Cores  
- Ajuste de Brilho  
- Ajuste de Contraste  
- Ajuste de Saturação  
- Filtro Laplaciano  
- Adicionar **stickers** na posição clicada com o mouse;
- Operações matemáticas
  - Adição
  - Subtração Ponderada
  - Blending

### Salvamento
- Salva a imagem final (com todos os efeitos aplicados) como `output_image.png`.

## Tecnologias Utilizadas

- **Python 3.10+**
- **OpenCV** (`opencv-python`)
- **Dear PyGui**
- **NumPy**

## Estrutura do Repositório

```plaintext
📂 GB_PG2025-2/
├── 📂 res/
│   ├── 📂 stickers/
│   │   ├── chocado.png
│   │   ├── tubarao.png
│   │   └── gato.png
│   └── colored_pencils_colour_pencils.jpg
├── 📂 src/      
│   └── EditorDeImagens.py
└── 📄 README.md
```
A pasta `res/stickers/` deve conter os arquivos PNG dos stickers utilizados no efeito “Sticker”.

## Como Executar

### Instale as dependências:
```bash
pip install opencv-python dearpygui
```

### Execute o programa:
```bash
python src/EditorDeImagens.py
```

## Como Usar 

- Abra o programa e escolha o modo:
   - Imagem
   - Câmera
- Clique em Selecionar Imagem para carregar um arquivo (caso selecione o modo imagem).
- Escolha um efeito no menu de efeitos:
- O controle correspondente (checkbox, slider ou combo) aparecerá automaticamente.
- Ajuste os valores conforme necessário.
- Para remover um filtro clique no botão de remover filtro
- Para aplicar stickers, inclua a opção de filtro "sticker" e então selecione o sticker desejado e clique na tela para posicioná-lo.
- Clique em Salvar Imagem para gerar o resultado final.
