from transformers import pipeline
from openai import OpenAI
import matplotlib.pyplot as plt
import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    KDPM2AncestralDiscreteScheduler,
    AutoencoderKL
)

def Huggingface_QeA():
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

def HuggingFace_Image(prompt):
    # Load VAE component
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float32
    )

    # Configure the pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "dataautogpt3/ProteusV0.1",
        vae=vae,
        torch_dtype=torch.float32
    )
    pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to('cpu')

    # Define prompts and generate image
    negative_prompt = "nsfw, bad quality, bad anatomy, worst quality, low quality, low resolutions, extra fingers, blur, blurry, ugly, wrongs proportions, watermark, image artifacts, lowres, ugly, jpeg artifacts, deformed, noisy image"

    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=1024,
        guidance_scale=7,
        num_inference_steps=20
    ).images[0]

    plt.imshow(image)
    plt.show()

def GPT_Text():
    cliente = OpenAI(api_key='')

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

def Diffusion(prompt):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, variant="fp16")
    pipe.to("cpu")


    # if using torch < 2.0
    # pipe.enable_xformers_memory_efficient_attention()

    images = pipe(prompt=prompt).images[0]

    plt.imshow(images)
    plt.show()

Diffusion('gatinho de olhos azuis')

