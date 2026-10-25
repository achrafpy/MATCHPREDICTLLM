# MATCHPREDICTLLM

Modelo Python de probabilidades de marcador con presupuesto de **0 €**. No necesita
un LLM ni una API de pago. No hay una predicción real guardada: todavía faltan datos
verificados del encuentro Barcelona–Real Madrid indicado para el 25/10/2026.
El horario y la sede comunicados por el usuario no se han confirmado.

## Qué calcula

Poisson independiente con ataque y defensa por equipo, ventaja local, forma de los
cinco últimos partidos y regularización L2. Incluye todos los rivales presentes en
los datos, no solamente enfrentamientos directos. Pondera más los partidos recientes
(semivida de 30 días) y utiliza exclusivamente los dos meses naturales anteriores
al instante `--as-of`. No usa resultados de partidos todavía en curso.

Con datos completos incorpora regates de equipo; con estadísticas de jugador y
alineaciones previstas históricas incorpora goles y regates recientes por minuto.
Los coeficientes se estiman con los datos: no se presupone que hacer más regates
cause más goles. Un identificador estable de jugador permite usar su rendimiento
en el club anterior, mientras la alineación prevista lo asigna al equipo del partido.
No descubre traspasos ni lesiones automáticamente.

Es un modelo experimental con supuestos explícitos, **no un modelo profesional
validado**. Dos meses pueden ser insuficientes para estimar fuerzas y aportaciones
de jugadores. No incluye una corrección Dixon–Coles, intervalos de incertidumbre,
calibración comprobada ni ajustes automáticos por expulsiones, lesiones o fuerza
de cada competición. La regularización (8) y semivida no se han optimizado con datos
reales. Mezclar Liga y Champions requiere resultados suficientes de sus rivales;
los grupos de equipos desconectados no permiten comparaciones fiables.

## Instalación y comprobación

Python 3.12, en el checkout existente; no hace falta crear un worktree.

```bash
cd /workspace/MATCHPREDICTLLM
python -m venv .venv
PIP_CACHE_DIR=/workspace/.cache/pip .venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

## Fuente gratuita de resultados

```bash
.venv/bin/python -m matchpredict download --season 2627 --league SP1 --output data/laliga.csv
```

Por defecto descarga los resultados comunitarios de
[OpenFootball](https://github.com/openfootball/football.json/blob/master/2026-27/es.1.json)
desde `raw.githubusercontent.com`, guardando URL, fecha de descarga y SHA-256.
El acceso se ha probado: el archivo tiene 64 resultados y el último es del
**20/09/2026**; por tanto está atrasado para el encuentro solicitado. Es una fuente
comunitaria que no se ha contrastado independientemente. Se excluyen cinco entradas
cuyo campo `score` no identifica explícitamente un resultado final `ft`.

También existe `--source football-data`, que descarga el CSV público de
football-data.co.uk. Ese acceso devolvió un bloqueo de red 403 en este entorno;
no se ha probado una descarga real de esa fuente.
Ambos feeds proporcionan resultados, **no** regates, alineaciones ni Champions.
Para evitar suponer una zona horaria, se importa la fecha con precisión diaria y
se retrasa el uso del resultado hasta la segunda medianoche UTC siguiente.
Esto excluye los resultados más recientes hasta 48 horas y limita la precisión de
los bordes de la ventana. Para análisis intradía usa datos con horas verificadas.

No se sobrescriben archivos existentes. Puedes descargar manualmente resultados
desde una fuente permitida y convertirlos al formato siguiente.
Respeta las condiciones de uso de cada proveedor. No se han verificado aquí fuentes
gratuitas completas de regates y jugadores para agosto–octubre de 2026.

## Archivos de entrada

`matches.csv`: una fila por partido terminado, sin duplicados, con estas columnas:

```csv
match_id,kickoff,completed_at,competition,home_team,away_team,home_goals,away_goals,home_dribbles,away_dribbles
```

Timestamps ISO 8601 con zona horaria. `completed_at` es el momento desde el que el
resultado estaba disponible; no puede preceder al final real. Usa nombres de equipo
consistentes entre competiciones. Los goles son enteros no negativos; usa partidos
de 90 minutos más descuento (sin prórroga ni tandas). Los regates son **regates
completados**, con una definición uniforme. Ambas columnas de regates pueden
omitirse; si se suministran, deben estar completas en toda la ventana de entrenamiento.
Liga y Champions pueden estar en el mismo archivo, distinguidas por `competition`.

Para jugadores, suministra ambos archivos con `--players` y `--lineups`:

```csv
match_id,team,player_id,minutes,goals,dribbles
```

```csv
match_id,team,player_id,expected_minutes,known_at
```

El primero contiene estadísticas finales individuales de partidos de `matches.csv`.
El segundo contiene **estimaciones hechas antes del partido**, también para cada
partido histórico de entrenamiento y para el encuentro objetivo (`match_id=target`
por defecto). No uses minutos reales posteriores al partido como si fueran previstos:
eso contaminaría la evaluación. `known_at` debe ser anterior al saque inicial y al
corte de la predicción. Debe haber una sola fila por jugador y equipo/partido, con
minutos esperados entre 0 y 90 y un total de 990 por equipo, incluyendo suplentes
si corresponde. Usa el mismo `player_id` tras un traspaso. Los jugadores con pocos
minutos se aproximan a una tasa media con 180 minutos de regularización; no se
puede asumir conocida su aportación. Las estadísticas se consideran disponibles
al finalizar el partido: ajusta `completed_at` si su publicación fue posterior.

## Predicción y evaluación

Ejemplo **de comando**, no confirmación del calendario. A las 21:00 en Madrid del
25/10/2026 corresponden las 20:00 UTC. Usa el corte real y nunca una hora futura.

```bash
.venv/bin/python -m matchpredict predict \
  --matches data/laliga.csv --home 'FC Barcelona' --away 'Real Madrid CF' \
  --as-of 2026-10-25T18:00:00Z --kickoff 2026-10-25T21:00:00+01:00

.venv/bin/python -m matchpredict backtest \
  --matches data/laliga.csv --as-of 2026-10-25T18:00:00Z
```

Requiere al menos 40 partidos en la ventana y tres de cada equipo objetivo.
Rechaza datos si cualquiera de los dos equipos no tiene un resultado disponible
en los últimos 14 días. Es una salvaguarda de actualidad, no una prueba de cobertura
completa: comprueba también que no falten partidos o competiciones.
El JSON
incluye goles esperados, probabilidades 1X2, cinco marcadores más probables y datos
que faltan. El marcador modal puede tener una probabilidad muy baja: no equivale al
resultado final. Sin datos reales no se emite una cifra de ejemplo como pronóstico.

El backtest vuelve a entrenar antes de cada partido de las últimas 20 observaciones
(mínimo 60 registros). Compara la log-verosimilitud del marcador con una referencia
de medias local/visitante y da Brier 1X2. Informa cuántos partidos evaluó y excluyó.
Una evaluación pequeña o sintética no demuestra rentabilidad ni calibración.

## Estado de validación

Las 13 pruebas automatizadas usan datos **sintéticos** para verificar cálculo,
separación temporal, validación de entradas y tratamiento de traspasos. No constituyen
una predicción del Barcelona–Real Madrid ni una evaluación con resultados reales.
La importación real de OpenFootball se ha ejecutado. El pronóstico con ese archivo
se rechaza por falta de actualidad. No se ha conseguido validar un pronóstico real
que cumpla los requisitos de Liga, Champions, regates y plantillas actuales.

También se ejecutó un backtest histórico con el archivo comunitario, con corte
24/09/2026: **17 partidos evaluados y 3 excluidos** por insuficiente historial.
Log-loss de marcador 3,1934 frente a 3,3567 de la referencia y Brier 1X2 0,5756.
Son observaciones de una muestra pequeña y de una fuente no contrastada; no
demuestran precisión para el partido de octubre ni superioridad estadística.
