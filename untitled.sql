
CREATE TABLE leyes (
	ley varchar(100),
	organismo_sanc varchar(100)
	concepto varchar(255),
	sancion_min int,
	sancion_max int,
	sancion_adicional varchar(255)
);

.separator ","
.import sanciones.csv leyes
