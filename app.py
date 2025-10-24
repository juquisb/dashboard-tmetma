import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime

class RelatorioTMETMA:
    def __init__(self, caminho_excel):
        self.caminho_excel = caminho_excel
        self.filas_monitoradas = [
            'Acolhimento',
            'Tri-agora',
            'Tri-algumas_horas',
            'plantao',
            'Parcerias',
            'N2-Administrativo'
        ]
        self.metas = {
            'Acolhimento': {'TME': 3.0, 'TMA': 30.0},
            'Tri-agora': {'TME': 5.0, 'TMA': 60.0},
            'Tri-algumas_horas': {'TME': 60.0, 'TMA': 120.0},
            'plantao': {'TME': 5.0, 'TMA': 60.0},
            'Parcerias': {'TME': 3.0, 'TMA': 30.0},
            'N2-Administrativo': {'TME': 3.0, 'TMA': 30.0}
        }

    def ler_dados_excel(self):
        df = pd.read_excel(self.caminho_excel)
        df['Inicio da ação'] = pd.to_datetime(df['Inicio da ação'])
        return df

    def converter_tempo_para_minutos(self, tempo_obj):
        """Converte tempo (HH:MM:SS, datetime.time ou número) para minutos decimais."""
        try:
            if isinstance(tempo_obj, (int, float)):
                return float(tempo_obj)
            if pd.isna(tempo_obj):
                return 0.0
            if isinstance(tempo_obj, dt.time):
                return tempo_obj.hour * 60 + tempo_obj.minute + tempo_obj.second / 60
            if isinstance(tempo_obj, str) and ':' in tempo_obj:
                h, m, s = [int(x) for x in tempo_obj.split(':')]
                return h * 60 + m + s / 60
            return float(tempo_obj)
        except:
            return 0.0

    def formatar_tempo_hhmmss(self, minutos):
        """Converte minutos decimais para o formato hh:mm:ss."""
        if minutos <= 0 or pd.isna(minutos):
            return "00:00:00"
        total_segundos = int(round(minutos * 60))
        horas = total_segundos // 3600
        minutos_restantes = (total_segundos % 3600) // 60
        segundos = total_segundos % 60
        return f"{horas:02d}:{minutos_restantes:02d}:{segundos:02d}"

    def calcular_medias_por_data(self, df):
        """Agrupa os dados por data e calcula médias de TME/TMA por fila."""
        df['Data'] = df['Inicio da ação'].dt.date
        resultados_por_data = {}

        for data in sorted(df['Data'].unique()):
            dados_do_dia = df[df['Data'] == data]
            resultados_data = {}

            for fila in self.filas_monitoradas:
                dados_fila = dados_do_dia[dados_do_dia['Fila'] == fila]
                if dados_fila.empty:
                    resultados_data[fila] = {'TME': 0, 'TMA': 0, 'volume': 0}
                    continue

                tempos_fila = dados_fila['Tempo na Fila'].apply(self.converter_tempo_para_minutos)
                tempos_atendimento = dados_fila['Tempo de atendimento'].apply(self.converter_tempo_para_minutos)

                resultados_data[fila] = {
                    'TME': tempos_fila.mean(),
                    'TMA': tempos_atendimento.mean(),
                    'volume': len(dados_fila)
                }

            resultados_por_data[data] = resultados_data
        return resultados_por_data

    def gerar_graficos_por_fila(self, resultados_por_data):
        """Gera gráficos de TME e TMA por fila com dois eixos verticais separados."""
        for fila in self.filas_monitoradas:
            datas = []
            datas_formatadas = []
            tmes = []
            tmas = []
            volumes = []

            for data, resultados in resultados_por_data.items():
                if fila in resultados and resultados[fila]['volume'] > 0:
                    datas.append(data)
                    # Formata a data como "10/out"
                    datas_formatadas.append(data.strftime("%d/%b").lower())
                    tmes.append(resultados[fila]['TME'])
                    tmas.append(resultados[fila]['TMA'])
                    volumes.append(resultados[fila]['volume'])

            if not datas:
                continue

            # Criar figura com dois eixos y
            fig, ax1 = plt.figure(figsize=(12, 6)), plt.gca()
            ax2 = ax1.twinx()

            # Configurar cores
            cor_tme = 'green'
            cor_tma = 'blue'
            cor_meta_tme = 'lightgreen'
            cor_meta_tma = 'lightblue'

            # Plotar TME no eixo esquerdo (ax1)
            linha_tme = ax1.plot(datas, tmes, color=cor_tme, marker='o', linewidth=2, 
                               label='TME Real', markersize=6)
            # Linha da meta TME
            meta_tme_valor = self.metas[fila]['TME']
            linha_meta_tme = ax1.axhline(y=meta_tme_valor, color=cor_meta_tme, linestyle='--', 
                                       linewidth=2, label=f'Meta TME ({self.formatar_tempo_hhmmss(meta_tme_valor)})')

            # Plotar TMA no eixo direito (ax2)
            linha_tma = ax2.plot(datas, tmas, color=cor_tma, marker='s', linewidth=2, 
                               label='TMA Real', markersize=6)
            # Linha da meta TMA
            meta_tma_valor = self.metas[fila]['TMA']
            linha_meta_tma = ax2.axhline(y=meta_tma_valor, color=cor_meta_tma, linestyle='--', 
                                       linewidth=2, label=f'Meta TMA ({self.formatar_tempo_hhmmss(meta_tma_valor)})')

            # Rótulos de dados para TME
            for x, y in zip(datas, tmes):
                if y > 0:
                    tempo_formatado = self.formatar_tempo_hhmmss(y)
                    ax1.text(x, y + (max(tmes) * 0.02), tempo_formatado, ha='center', 
                           va='bottom', fontsize=9, color=cor_tme, fontweight='bold')

            # Rótulos de dados para TMA
            for x, y in zip(datas, tmas):
                if y > 0:
                    tempo_formatado = self.formatar_tempo_hhmmss(y)
                    ax2.text(x, y + (max(tmas) * 0.02), tempo_formatado, ha='center', 
                           va='bottom', fontsize=9, color=cor_tma, fontweight='bold')

            # Configurar eixo esquerdo (TME)
            ax1.set_xlabel('Data', fontsize=12, fontweight='bold')
            ax1.set_ylabel('TME (minutos)', color=cor_tme, fontsize=12, fontweight='bold')
            ax1.tick_params(axis='y', labelcolor=cor_tme)
            ax1.grid(True, linestyle='--', alpha=0.3, axis='both')
            
            # Configurar escala do eixo TME para melhor visualização
            if tmes:
                margem_tme = max(tmes) * 0.15
                ax1.set_ylim(0, max(tmes) + margem_tme)

            # Configurar eixo direito (TMA)
            ax2.set_ylabel('TMA (minutos)', color=cor_tma, fontsize=12, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor=cor_tma)
            
            # Configurar escala do eixo TMA para melhor visualização
            if tmas:
                margem_tma = max(tmas) * 0.15
                ax2.set_ylim(0, max(tmas) + margem_tma)

            # Configurar eixo x
            plt.xticks(ticks=datas, labels=datas_formatadas, rotation=45)
            
            # Título e legendas
            plt.title(f'{fila.upper()} - Desempenho TME e TMA\n(Eixos Separados para Melhor Visualização)', 
                     fontsize=14, fontweight='bold', pad=20)
            
            # Combinar legendas de ambos os eixos
            linhas, labels = ax1.get_legend_handles_labels()
            linhas2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(linhas + linhas2, labels + labels2, loc='upper right', 
                      bbox_to_anchor=(0, 1), frameon=True, fancybox=True)

            # Adicionar informação de volume no gráfico
            volume_total = sum(volumes)
            ax1.text(0.02, 0.98, f'Volume Total: {volume_total} atendimentos', 
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()
            plt.show()

            # Gráfico adicional: Volume por dia
            plt.figure(figsize=(10, 4))
            bars = plt.bar(datas_formatadas, volumes, color='orange', alpha=0.7)
            plt.title(f'{fila.upper()} - Volume de Atendimentos por Dia', fontsize=12, fontweight='bold')
            plt.xlabel('Data')
            plt.ylabel('Quantidade de Atendimentos')
            
            # Adicionar valores nas barras
            for bar, volume in zip(bars, volumes):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(volume), ha='center', va='bottom', fontweight='bold')
            
            plt.grid(True, linestyle='--', alpha=0.3, axis='y')
            plt.tight_layout()
            plt.show()

    def executar(self):
        print("🚀 Iniciando processamento de relatórios TMETMA...")

        df = self.ler_dados_excel()
        resultados = self.calcular_medias_por_data(df)
        self.gerar_graficos_por_fila(resultados)

        print("✅ Gráficos gerados com sucesso!")


# Exemplo de uso
if __name__ == "__main__":
    CAMINHO_EXCEL = r"C:\Users\Julio Bueno\Downloads\TMA_ANALISTAS.xlsx"
    analisador = RelatorioTMETMA(CAMINHO_EXCEL)
    analisador.executar()
