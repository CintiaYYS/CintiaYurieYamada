from codigo_fonte.processamento.csv_utilidades import postsComURL
from codigo_fonte.processamento.imagens_utilidades import comparar_todas_imagens_recorte
from codigo_fonte.scrapping.facebook_scraper_playwright import baixarImagens

if __name__ == "__main__":
    postsComURL()
    baixarImagens()
    #comparar_todas_imagens_recorte()