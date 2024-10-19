# Tech Challenge FIAP - Análise de Vídeo

## Descrição do Projeto

O Tech Challenge desta fase será a criação de uma aplicação que utilize **análise de vídeo**. O seu projeto deve incorporar as técnicas de reconhecimento facial, análise de expressões emocionais em vídeos e detecção de atividades.

## Proposta do Desafio

Você deverá criar uma aplicação a partir do vídeo que se encontra disponível na plataforma do aluno, e que execute as seguintes tarefas:
1. Reconhecimento facial: Identifique e marque os rostos presentes no vídeo.
2. Análise de expressões emocionais: Analise as expressões emocionais dos rostos identificados.
3. Detecção de atividades: Detecte e categorize as atividades sendo realizadas no vídeo.
4. Geração de resumo: Crie um resumo automático das principais atividades e emoções detectadas no vídeo.

### Bibliotecas Utilizadas

- **OpenCV**: Biblioteca amplamente utilizada para processamento de imagens e vídeos.
- **MediaPipe**: Framework de aprendizado de máquina focado em detecção facial e rastreamento de pontos faciais (landmarks).
- **Deepface**: Biblioteca amplamente utilizada para identificação de sentimentos em imagens e vídeos.
- **YOLOv5**: Modelo utilizado para identificação de objetos em imagens.

## Requisitos

Antes de rodar o projeto, certifique-se de que os seguintes requisitos estão instalados:

- **Python 3.7 ou superior**
- **pip** (gerenciador de pacotes do Python)

### Instalação das Dependências

Para instalar as bibliotecas necessárias, execute o comando abaixo:

```bash
pip install opencv-python mediapipe deepface tqdm tf-keras
```

## Como Executar

1. **Clone o repositório** para sua máquina local ou faça o download do código.

2. **Tenha um vídeo** para ser executado pelo modelo. 

3. **Atualize a função main** com o diretório onde está o vídeo alvo e onde deve ser registrado o output.
   
4. **Execute o script principal** com o seguinte comando:

```bash
python facial_recognition.py
```

## Exemplos de Uso

Ao rodar o script, no terminal será exibida uma barra de progresso com o status do script. Ao final será gerado um arquivo de vídeo com o resultado do script.


## Referências

- [Documentação OpenCV](https://pypi.org/project/opencv-python/)
- [Documentação MediaPipe](https://pypi.org/project/mediapipe/)
- [Documentação Deepface](https://pypi.org/project/deepface/)
- [Documentação YOLO](https://pytorch.org/hub/ultralytics_yolov5/)
