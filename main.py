from transformers import pipeline
from openai import OpenAI

def Huggingface():
    #Definindo modelo pré-treinado baseado em perguntas e respostas - https://huggingface.co/models?sort=trending
    qa_model = pipeline("question-answering", "timpal0l/mdeberta-v3-base-squad2")

    #Definindo pergunta e contexto
    pergunta = "Consigo escapar de um buraco negro?"
    contexto = "Buraco negro é uma região do espaço-tempo em que o campo gravitacional é tão intenso que nada nenhuma partícula ou radiação eletromagnética como a luz pode escapar. A teoria da relatividade geral prevê que uma massa suficientemente compacta pode deformar o espaço-tempo para formar um buraco negro. O limite da região da qual não é possível escapar é chamado de horizonte de eventos. Embora o horizonte de eventos tenha um efeito enorme sobre o destino e as circunstâncias de um objeto que o atravessa, não tem nenhuma característica local detectável. De muitas maneiras, um buraco negro age como um corpo negro ideal, pois não reflete luz. Além disso, a teoria quântica de campos no espaço-tempo curvo prevê que os horizontes de eventos emitem radiação Hawking, com o mesmo espectro que um corpo negro de temperatura inversamente proporcional à sua massa. Essa temperatura é da ordem dos bilionésimos de um kelvin para buracos negros de massa estelar, o que a torna praticamente impossível de observar."

    #Obtendo resposta
    resposta = qa_model(question = pergunta, context = contexto)

    #Mostrando resultados
    print('Pergunta:'+pergunta)
    print('Resposta:'+resposta['answer'])
    print('Score:'+str(resposta['score']))

def GPT_Text():
    cliente = OpenAI(api_key='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')

    response = cliente.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Consigo escapar de um buraco negro?",
            }
        ],
        model="text-embedding-ada-002",
        max_tokens=80
    )

    print(response.choices[0].text)

Huggingface()

