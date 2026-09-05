-- =====================================================
-- VIEWS PARA ANALÍTICA Y POWER BI
-- =====================================================


-- 1. Resumen mensual de producción
CREATE VIEW vw_produccion_mensual AS
SELECT
    anio,
    mes,
    SUM(prod_pet) AS produccion_petroleo,
    SUM(prod_gas) AS produccion_gas,
    SUM(prod_agua) AS produccion_agua,
    SUM(iny_agua) AS inyeccion_agua,
    SUM(iny_gas) AS inyeccion_gas
FROM produccion
GROUP BY anio, mes;


-- 2. Resumen por provincia
CREATE VIEW vw_produccion_provincia AS
SELECT
    provincia,
    COUNT(DISTINCT idpozo) AS cantidad_pozos,
    SUM(prod_pet) AS produccion_petroleo,
    SUM(prod_gas) AS produccion_gas,
    SUM(prod_agua) AS produccion_agua,
    AVG(prod_pet) AS promedio_petroleo,
    AVG(prod_gas) AS promedio_gas
FROM produccion
GROUP BY provincia;


-- 3. Resumen por cuenca
CREATE VIEW vw_produccion_cuenca AS
SELECT
    cuenca,
    COUNT(DISTINCT idpozo) AS cantidad_pozos,
    SUM(prod_pet) AS produccion_petroleo,
    SUM(prod_gas) AS produccion_gas,
    SUM(prod_agua) AS produccion_agua
FROM produccion
GROUP BY cuenca;


-- 4. Resumen por empresa
CREATE VIEW vw_produccion_empresa AS
SELECT
    empresa,
    COUNT(DISTINCT idpozo) AS cantidad_pozos,
    SUM(prod_pet) AS produccion_petroleo,
    SUM(prod_gas) AS produccion_gas,
    AVG(prod_pet) AS promedio_petroleo,
    AVG(prod_gas) AS promedio_gas
FROM produccion
GROUP BY empresa;


-- 5. Resumen por tipo de extracción
CREATE VIEW vw_tipo_extraccion AS
SELECT
    tipoextraccion,
    COUNT(DISTINCT idpozo) AS cantidad_pozos,
    SUM(prod_pet) AS produccion_petroleo,
    SUM(prod_gas) AS produccion_gas
FROM produccion
GROUP BY tipoextraccion;


-- 6. Resumen por tipo de recurso
CREATE VIEW vw_tipo_recurso AS
SELECT
    tipo_de_recurso,
    COUNT(DISTINCT idpozo) AS cantidad_pozos,
    SUM(prod_pet) AS produccion_petroleo,
    SUM(prod_gas) AS produccion_gas
FROM produccion
GROUP BY tipo_de_recurso;


-- 7. Resumen por pozo
CREATE VIEW vw_produccion_pozo AS
SELECT
    idpozo,
    provincia,
    cuenca,
    empresa,
    tipopozo,
    tipoextraccion,
    MAX(profundidad) AS profundidad,
    SUM(prod_pet) AS produccion_petroleo,
    SUM(prod_gas) AS produccion_gas,
    AVG(tef) AS tef_promedio
FROM produccion
GROUP BY
    idpozo,
    provincia,
    cuenca,
    empresa,
    tipopozo,
    tipoextraccion;