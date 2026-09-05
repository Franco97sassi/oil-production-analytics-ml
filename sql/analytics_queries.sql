-- =====================================================
-- YPF DATA ANALYTICS
-- Consultas analíticas sobre producción hidrocarburífera
-- =====================================================


-- 1. Producción total de petróleo
SELECT
    SUM(prod_pet) AS produccion_total_petroleo
FROM produccion;


-- 2. Producción total de gas
SELECT
    SUM(prod_gas) AS produccion_total_gas
FROM produccion;


-- 3. Producción total de agua
SELECT
    SUM(prod_agua) AS produccion_total_agua
FROM produccion;


-- 4. Producción de petróleo por año
SELECT
    anio,
    SUM(prod_pet) AS produccion_petroleo
FROM produccion
GROUP BY anio
ORDER BY anio;


-- 5. Producción de gas por año
SELECT
    anio,
    SUM(prod_gas) AS produccion_gas
FROM produccion
GROUP BY anio
ORDER BY anio;


-- 6. Producción de petróleo por provincia
SELECT
    provincia,
    SUM(prod_pet) AS produccion_petroleo
FROM produccion
GROUP BY provincia
ORDER BY produccion_petroleo DESC;


-- 7. Producción de petróleo por cuenca
SELECT
    cuenca,
    SUM(prod_pet) AS produccion_petroleo
FROM produccion
GROUP BY cuenca
ORDER BY produccion_petroleo DESC;


-- 8. Producción de gas por cuenca
SELECT
    cuenca,
    SUM(prod_gas) AS produccion_gas
FROM produccion
GROUP BY cuenca
ORDER BY produccion_gas DESC;


-- 9. Top 10 empresas por producción de petróleo
SELECT
    empresa,
    SUM(prod_pet) AS produccion_petroleo
FROM produccion
GROUP BY empresa
ORDER BY produccion_petroleo DESC
LIMIT 10;


-- 10. Top 10 pozos por producción de petróleo
SELECT
    idpozo,
    provincia,
    cuenca,
    SUM(prod_pet) AS produccion_petroleo
FROM produccion
GROUP BY idpozo, provincia, cuenca
ORDER BY produccion_petroleo DESC
LIMIT 10;


-- 11. Producción según tipo de extracción
SELECT
    tipoextraccion,
    SUM(prod_pet) AS produccion_petroleo,
    SUM(prod_gas) AS produccion_gas
FROM produccion
GROUP BY tipoextraccion
ORDER BY produccion_petroleo DESC;


-- 12. Producción por tipo de pozo
SELECT
    tipopozo,
    SUM(prod_pet) AS produccion_petroleo,
    SUM(prod_gas) AS produccion_gas
FROM produccion
GROUP BY tipopozo
ORDER BY produccion_petroleo DESC;


-- 13. Producción según tipo de recurso
SELECT
    tipo_de_recurso,
    SUM(prod_pet) AS produccion_petroleo,
    SUM(prod_gas) AS produccion_gas
FROM produccion
GROUP BY tipo_de_recurso
ORDER BY produccion_petroleo DESC;


-- 14. Evolución mensual de producción
SELECT
    anio,
    mes,
    SUM(prod_pet) AS produccion_petroleo,
    SUM(prod_gas) AS produccion_gas,
    SUM(prod_agua) AS produccion_agua
FROM produccion
GROUP BY anio, mes
ORDER BY anio, mes;


-- 15. Promedio de producción por pozo
SELECT
    idpozo,
    AVG(prod_pet) AS promedio_petroleo,
    AVG(prod_gas) AS promedio_gas
FROM produccion
GROUP BY idpozo
ORDER BY promedio_petroleo DESC;


-- 16. Pozos con mayor profundidad promedio
SELECT
    idpozo,
    AVG(profundidad) AS profundidad_promedio
FROM produccion
WHERE profundidad IS NOT NULL
GROUP BY idpozo
ORDER BY profundidad_promedio DESC
LIMIT 10;


-- 17. Producción por formación
SELECT
    formacion,
    SUM(prod_pet) AS produccion_petroleo,
    SUM(prod_gas) AS produccion_gas
FROM produccion
WHERE formacion IS NOT NULL
GROUP BY formacion
ORDER BY produccion_petroleo DESC;


-- 18. Inyección total por tipo
SELECT
    SUM(iny_agua) AS inyeccion_agua,
    SUM(iny_gas) AS inyeccion_gas,
    SUM(iny_co2) AS inyeccion_co2,
    SUM(iny_otro) AS inyeccion_otro
FROM produccion;


-- 19. Producción promedio por provincia
SELECT
    provincia,
    AVG(prod_pet) AS promedio_petroleo,
    AVG(prod_gas) AS promedio_gas
FROM produccion
GROUP BY provincia
ORDER BY promedio_petroleo DESC;


-- 20. Cantidad de pozos únicos por provincia
SELECT
    provincia,
    COUNT(DISTINCT idpozo) AS cantidad_pozos
FROM produccion
GROUP BY provincia
ORDER BY cantidad_pozos DESC;