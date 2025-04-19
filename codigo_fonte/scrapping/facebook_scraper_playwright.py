from playwright.sync_api import sync_playwright
import requests
import os
import pandas as pd
import re
from urllib.parse import urlparse
from pathlib import Path
from PIL import Image
from io import BytesIO


def scrape_facebook_post(post_url):
    
    save_dir="dados/brutos/imagens"
    largura_min=300
    altura_min=300

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(post_url, timeout=20000)
        page.wait_for_timeout(5000)

        # Coletar todas as imagens
        image_urls = page.eval_on_selector_all("img", "elements => elements.map(e => ({src: e.src, width: e.naturalWidth, height: e.naturalHeight}))")
        browser.close()

        # Selecionar a maior imagem acima do limite definido
        image_url = None
        max_area = 0

        for img in image_urls:
            if img['width'] >= largura_min and img['height'] >= altura_min:
                area = img['width'] * img['height']
                if area > max_area:
                    max_area = area
                    image_url = img['src']

        # Baixar e salvar
        saved_image_path = None
        if image_url:
            try:
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                os.makedirs(save_dir, exist_ok=True)
                file_name = os.path.basename(urlparse(image_url).path)
                saved_image_path = os.path.join(save_dir, file_name)
                img.save(saved_image_path)
                return file_name
            except Exception as e: pass        

    return None




def baixarImagens():

    # Caminho do CSV
    csv_path = Path("dados/processados/base_posts.csv")

    # Expressão regular para extrair URLs de postagens do Facebook
    padrao_url = r'https?://www\.facebook\.com/[^ ]+'

    # Lê o arquivo CSV
    df = pd.read_csv(csv_path)

    # Extrai a primeira URL da coluna 'Content' e remove as que não têm URL
    df['url_post'] = df['Content'].astype(str).apply(lambda texto: re.findall(padrao_url, texto))
    df['url_post'] = df['url_post'].apply(lambda urls: urls[0] if urls else None)
    df = df.dropna(subset=['url_post']).reset_index(drop=True)

    # for url in lista_urls:
    #     scrape_facebook_post(url)
    
    # Lista para armazenar os nomes das imagens
    nomes_imagens = []

    # Itera nas URLs válidas
    for index, row in df.iterrows():
        url = row['url_post']
        nome_imagem = scrape_facebook_post(url)  # deve retornar None se falhar
        nomes_imagens.append(nome_imagem)

    # Adiciona a coluna com os nomes das imagens
    df['nome_imagem'] = nomes_imagens

    # Remove linhas onde a imagem não foi salva (nome_imagem == None)
    df = df.dropna(subset=['nome_imagem']).reset_index(drop=True)

    # Salva o DataFrame final no mesmo CSV (ou em outro, se preferir)
    df.to_csv(csv_path, index=False)