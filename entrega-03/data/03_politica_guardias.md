# Política de guardias e incidentes de Kairo

> Documento interno de Transportes Altiplano S.A. Aprobado por la Gerencia de Tecnología. Vigente desde enero de 2026.

## Rotación de guardias

Cada equipo dueño de un servicio de Kairo tiene una rotación de guardia semanal. La semana de guardia empieza el **miércoles a las 10:00** y termina el miércoles siguiente a la misma hora. Se eligió el miércoles para que el traspaso no coincida con el congelamiento de despliegues de los fines de semana.

Además de las guardias por equipo, existe el rol de **líder de incidentes**, que rota entre las personas con más de un año en la empresa. El líder de incidentes no arregla el problema técnico: coordina, comunica y decide escalamientos.

Las guardias se administran en la herramienta Sereno. Cada persona puede intercambiar su semana con otra, siempre que el cambio quede registrado en Sereno con al menos 48 horas de anticipación.

## Compensación

La semana de guardia se compensa con un adicional fijo equivalente al 8 % del salario mensual bruto. Además, cada hora trabajada fuera del horario laboral por un incidente se compensa con una hora libre, que se puede tomar dentro de los 60 días siguientes. Una persona no puede hacer más de una semana de guardia por mes salvo que lo pida explícitamente.

## Niveles de severidad

| Severidad | Definición | Tiempo de respuesta | Quién se convoca |
|---|---|---|---|
| SEV1 | Ningún centro de distribución puede despachar, o hay pérdida de datos de facturación | 10 minutos | Guardia del servicio, líder de incidentes y gerente de Tecnología |
| SEV2 | Un centro de distribución no puede operar, o una funcionalidad crítica está degradada para todos | 20 minutos | Guardia del servicio y líder de incidentes |
| SEV3 | Funcionalidad no crítica caída, o degradación con alternativa manual disponible | 4 horas hábiles | Guardia del servicio |
| SEV4 | Problema menor o cosmético | Próximo sprint | Nadie fuera del horario laboral |

El tiempo de respuesta se mide desde que suena la alerta hasta que alguien la reconoce en Sereno. Si la guardia principal no reconoce una alerta SEV1 o SEV2 en el tiempo indicado, Sereno escala automáticamente a la guardia secundaria y, cinco minutos después, al responsable del equipo.

## Cómo se declara un incidente

Cualquier persona de la empresa puede declarar un incidente escribiendo `/incidente` en el canal `#kairo-incidentes`. El comando crea un canal dedicado con el formato `inc-AAAAMMDD-servicio`, por ejemplo `inc-20260314-despacho`, y avisa al líder de incidentes de turno.

Durante un incidente SEV1 o SEV2:

- El líder de incidentes publica una actualización de estado **cada 30 minutos** en el canal del incidente, aunque no haya novedades.
- Si el incidente afecta a clientes corporativos, el equipo de Atención Comercial publica un aviso en la página de estado pública dentro de los primeros 15 minutos.
- No se despliegan cambios que no estén relacionados con la solución del incidente.

## Postmortems

Todo incidente SEV1 o SEV2 requiere un postmortem escrito dentro de los **cinco días hábiles** posteriores a su cierre. El postmortem sigue la plantilla del repositorio `kairo-postmortems` y es sin culpables: describe qué falló en los sistemas y procesos, no quién se equivocó.

Cada postmortem tiene que incluir:

1. Cronología del incidente, con hora de detección, de mitigación y de resolución.
2. Impacto en números: envíos demorados, clientes afectados y minutos de indisponibilidad.
3. Causa raíz y factores que contribuyeron.
4. Acciones correctivas, cada una con un responsable y una fecha.

Las acciones correctivas de un SEV1 tienen prioridad sobre cualquier trabajo planificado del sprint. Los postmortems se revisan en la reunión de confiabilidad, que se hace el primer jueves de cada mes.

## Incidente de referencia: el apagón de rastreo de 2025

El incidente más grave desde la puesta en marcha de Kairo ocurrió el 3 de septiembre de 2025. Una actualización de TimescaleDB dejó al servicio `rastreo` sin poder escribir posiciones durante 3 horas y 40 minutos. Los choferes siguieron trabajando, pero los clientes no pudieron ver dónde estaban sus envíos. Se clasificó como SEV2 porque `despacho` siguió funcionando.

Como resultado del postmortem se tomaron dos medidas: las actualizaciones de versión de bases de datos ahora se prueban primero en `kairo-staging` con una copia anonimizada de los datos de producción, y `rastreo` guarda las posiciones en una cola local por hasta dos horas si no puede escribir en la base.
