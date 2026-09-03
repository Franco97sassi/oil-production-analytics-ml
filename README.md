curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "mes": 10,
  "iny_agua": 0,
  "iny_gas": 0,
  "tef": 31,
  "tipoextraccion": "Bombeo Mecánico",
  "tipoestado": "Extracción Efectiva",
  "tipopozo": "Petrolífero",
  "provincia": "Santa Cruz",
  "cuenca": "GOLFO SAN JORGE",
  "prod_pet_lag1": 52.86
}'
Request URL
http://127.0.0.1:8000/predict
Server response
Code	Details
200	
Response body
Download
{
  "produccion_predicha": 54.91
}