DROP TABLE IF EXISTS produccion;

CREATE TABLE produccion (
    idempresa TEXT,
    anio INTEGER,
    mes INTEGER,
    idpozo TEXT,

    prod_pet REAL,
    prod_gas REAL,
    prod_agua REAL,

    iny_agua REAL,
    iny_gas REAL,
    iny_co2 REAL,
    iny_otro REAL,

    tef REAL,
    vida_util REAL,

    tipoextraccion TEXT,
    tipoestado TEXT,
    tipopozo TEXT,

    observaciones TEXT,
    fechaingreso DATE,

    rectificado TEXT,
    habilitado TEXT,

    idusuario TEXT,

    empresa TEXT,
    sigla TEXT,

    formprod TEXT,
    profundidad REAL,
    formacion TEXT,

    idareapermisoconcesion TEXT,
    areapermisoconcesion TEXT,

    idareayacimiento TEXT,
    areayacimiento TEXT,

    cuenca TEXT,
    provincia TEXT,

    tipo_de_recurso TEXT,
    proyecto TEXT,

    clasificacion TEXT,
    subclasificacion TEXT,
    sub_tipo_recurso TEXT,

    fecha_data DATE
);