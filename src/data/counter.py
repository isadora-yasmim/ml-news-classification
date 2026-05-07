import json
from collections import Counter
from collections import defaultdict

CAMINHO_DATASET = "././data/raw/News_Category_Dataset_v3.json"
ARQUIVO_SAIDA = "relatorio_dataset.md"

# ==========================================
# ESTRUTURAS AUXILIARES
# ==========================================

contador_categorias = Counter()
headlines = defaultdict(list)
linhas_nulas = []

total_registros = 0

# ==========================================
# LEITURA DO DATASET
# ==========================================

with open(CAMINHO_DATASET, "r", encoding="utf-8") as arquivo:

    for numero_linha, linha in enumerate(arquivo, start=1):

        if not linha.strip():
            continue

        total_registros += 1

        try:
            noticia = json.loads(linha)

            # ==========================
            # CONTAGEM DE CATEGORIAS
            # ==========================
            categoria = noticia.get("category", "SEM CATEGORIA")
            contador_categorias[categoria] += 1

            # ==========================
            # VERIFICAÇÃO DE DUPLICADOS
            # ==========================
            headline = noticia.get("headline")

            if headline:
                headlines[headline].append(numero_linha)

            # ==========================
            # VERIFICAÇÃO DE NULOS
            # ==========================
            campos_nulos = []

            for chave, valor in noticia.items():
                if valor is None or valor == "":
                    campos_nulos.append(chave)

            if campos_nulos:
                linhas_nulas.append({
                    "linha": numero_linha,
                    "campos_nulos": campos_nulos,
                    "conteudo": noticia
                })

        except json.JSONDecodeError:
            print(f"[ERRO] Linha {numero_linha} possui JSON inválido.")

# ==========================================
# GERAÇÃO DO RELATÓRIO
# ==========================================

with open(ARQUIVO_SAIDA, "w", encoding="utf-8") as relatorio:

    relatorio.write("# Relatório de Análise do Dataset\n\n")

    # ======================================
    # RESUMO
    # ======================================

    relatorio.write("## Resumo Geral\n\n")
    relatorio.write(f"- Total de registros analisados: **{total_registros}**\n")
    relatorio.write(f"- Total de categorias diferentes: **{len(contador_categorias)}**\n")

    total_duplicados = sum(
        1 for linhas in headlines.values()
        if len(linhas) > 1
    )

    relatorio.write(f"- Total de headlines duplicadas: **{total_duplicados}**\n")
    relatorio.write(f"- Total de linhas com valores nulos: **{len(linhas_nulas)}**\n\n")

    # ======================================
    # CATEGORIAS
    # ======================================

    relatorio.write("---\n\n")
    relatorio.write("## Categorias e Quantidade de Documentos\n\n")

    for categoria, quantidade in contador_categorias.most_common():
        relatorio.write(f"- {categoria}: {quantidade}\n")

    # ======================================
    # DUPLICADOS
    # ======================================

    relatorio.write("\n---\n\n")
    relatorio.write("## Textos Duplicados\n\n")

    duplicados_encontrados = False

    for headline, linhas in headlines.items():

        if len(linhas) > 1:

            duplicados_encontrados = True

            relatorio.write(f"### Headline Duplicada\n\n")
            relatorio.write(f"**Texto:**\n\n")
            relatorio.write(f"> {headline}\n\n")
            relatorio.write(f"**Linhas:** {linhas}\n\n")

    if not duplicados_encontrados:
        relatorio.write("Nenhum texto duplicado encontrado.\n")

    # ======================================
    # VALORES NULOS
    # ======================================

    relatorio.write("\n---\n\n")
    relatorio.write("## Valores Nulos\n\n")

    if linhas_nulas:

        for item in linhas_nulas:

            relatorio.write(f"### Linha {item['linha']}\n\n")
            relatorio.write(f"**Campos nulos:** {item['campos_nulos']}\n\n")
            relatorio.write("**Conteúdo do registro:**\n\n")
            relatorio.write("```json\n")
            relatorio.write(
                json.dumps(
                    item["conteudo"],
                    indent=4,
                    ensure_ascii=False
                )
            )
            relatorio.write("\n```\n\n")

    else:
        relatorio.write("Nenhum valor nulo encontrado.\n")

# ==========================================
# FINALIZAÇÃO
# ==========================================

print("=" * 60)
print("RELATÓRIO GERADO COM SUCESSO")
print("=" * 60)

print(f"\nArquivo salvo em: {ARQUIVO_SAIDA}")