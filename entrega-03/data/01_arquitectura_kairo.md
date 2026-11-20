# Kairo: arquitectura general de la plataforma

> Documento interno de Transportes Altiplano S.A. Equipo de Plataforma. Última revisión: marzo de 2026.

## Qué es Kairo

Kairo es la plataforma interna que usa Transportes Altiplano para cotizar, despachar y seguir los envíos de carga entre sus 14 centros de distribución. Reemplazó en 2024 al sistema anterior, llamado Tambo, que era un monolito en PHP con una sola base de datos MySQL. La migración se hizo en tres etapas y terminó en noviembre de 2024, cuando se apagó definitivamente el último servidor de Tambo en el centro de datos de Salta.

La plataforma atiende a tres tipos de usuarios: los operadores de los centros de distribución, los clientes corporativos que integran sus sistemas por API y los choferes, que usan la aplicación móvil Ruta.

## Servicios principales

Kairo está dividido en seis servicios. Cada uno tiene un equipo dueño y su propio repositorio.

| Servicio | Responsabilidad | Lenguaje | Equipo dueño |
|---|---|---|---|
| `cotizador` | Calcula tarifas de envío según peso, volumen, distancia y urgencia | Kotlin | Equipo Tarifas |
| `despacho` | Asigna envíos a camiones y arma las hojas de ruta | Go | Equipo Operaciones |
| `rastreo` | Recibe posiciones GPS de los camiones y publica eventos de seguimiento | Go | Equipo Operaciones |
| `portal` | Frontend web para operadores y clientes | TypeScript (Angular) | Equipo Experiencia |
| `facturador` | Emite facturas electrónicas y notas de crédito | Kotlin | Equipo Finanzas Técnicas |
| `notificador` | Envía avisos por correo, SMS y push | Python | Equipo Experiencia |

El servicio más crítico es `despacho`: si se cae, los centros de distribución no pueden cargar camiones. Por eso tiene tres réplicas como mínimo en producción, mientras que el resto de los servicios corre con dos.

## Comunicación entre servicios

Los servicios se comunican de dos maneras:

- **Sincrónica**, por gRPC, solo para consultas que necesitan respuesta inmediata. Por ejemplo, `portal` consulta a `cotizador` cuando un cliente pide un presupuesto en pantalla.
- **Asincrónica**, mediante eventos publicados en NATS JetStream. Todos los cambios de estado de un envío (creado, asignado, en viaje, entregado, rechazado) se publican en el stream `envios.estado`. Los consumidores guardan el último evento procesado para poder reanudar si se reinician.

La regla del equipo de Plataforma es que ningún servicio puede leer directamente la base de datos de otro. Si un servicio necesita datos ajenos, los pide por gRPC o mantiene una copia local alimentada por eventos.

## Almacenamiento

Cada servicio tiene su propia base de datos:

- `cotizador`, `despacho` y `facturador` usan PostgreSQL 16, en instancias separadas.
- `rastreo` guarda las posiciones GPS en TimescaleDB, porque recibe unas 40.000 posiciones por minuto en horario pico y necesita consultas por rango de tiempo.
- `notificador` no tiene base de datos propia: guarda el historial de envíos de mensajes en un bucket de almacenamiento de objetos, en archivos Parquet particionados por día.

Las copias de seguridad de PostgreSQL se hacen cada seis horas y se retienen 35 días. Las de TimescaleDB se hacen una vez por día y se retienen 14 días, porque los datos de posiciones con más de dos semanas se agregan por hora y los datos crudos se descartan.

## Infraestructura

Kairo corre sobre un clúster de Kubernetes administrado, en dos zonas de disponibilidad. El clúster de producción se llama `kairo-prod` y el de pruebas `kairo-staging`. Hay un tercer clúster, `kairo-sandbox`, que los clientes corporativos usan para probar sus integraciones sin afectar datos reales; se borra y se recrea todos los domingos a las 03:00.

La configuración de infraestructura vive en el repositorio `kairo-infra` y se aplica con Terraform. Los manifiestos de Kubernetes se sincronizan con Argo CD desde el repositorio `kairo-deploy`.

## Observabilidad

Todos los servicios exponen métricas en formato Prometheus en el puerto 9102 y escriben logs en JSON a la salida estándar. Las trazas distribuidas se envían con OpenTelemetry. Los tableros principales están en Grafana, en la carpeta "Kairo / Producción".

Cada servicio debe definir al menos dos indicadores de nivel de servicio (SLI): la tasa de errores y la latencia del percentil 99. El objetivo de disponibilidad de `despacho` es 99,95 % mensual; el del resto de los servicios es 99,9 %.

## Decisiones de arquitectura

Las decisiones importantes se registran como ADR (Architecture Decision Records) en la carpeta `docs/adr` del repositorio `kairo-infra`. Algunas de las más consultadas:

- **ADR-007**: por qué se eligió NATS JetStream en lugar de Kafka. El motivo principal fue el costo operativo: el equipo de Plataforma tenía solo tres personas y NATS requería menos mantenimiento.
- **ADR-012**: por qué `rastreo` se escribió en Go y no en Kotlin. Las pruebas de carga mostraron que la versión en Go usaba un 60 % menos de memoria con el mismo volumen de posiciones.
- **ADR-019**: prohibición de lecturas cruzadas entre bases de datos de distintos servicios.
