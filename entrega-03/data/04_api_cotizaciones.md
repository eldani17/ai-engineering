# API pública de cotizaciones de Kairo

> Documentación para clientes corporativos de Transportes Altiplano S.A. Versión de la API: v3.

## Introducción

La API de cotizaciones permite a los clientes corporativos calcular el precio de un envío y reservarlo desde sus propios sistemas. Está implementada por el servicio `cotizador` y se publica en `https://api.altiplano.example/v3`. La versión v2 de la API dejará de funcionar el **30 de junio de 2026**; los clientes que todavía la usan recibieron un aviso por correo.

Para probar integraciones sin generar envíos reales se usa el ambiente sandbox, en `https://sandbox.altiplano.example/v3`. Los datos del sandbox se borran todos los domingos a las 03:00.

## Autenticación

Cada cliente recibe un par de credenciales: un identificador de cliente y una clave secreta. Con ellas se obtiene un token de acceso en el endpoint `POST /oauth/token`, usando el flujo client credentials de OAuth 2.0. El token dura **60 minutos** y se envía en el encabezado `Authorization: Bearer <token>`.

Las claves secretas vencen cada 180 días. Quince días antes del vencimiento, el contacto técnico del cliente recibe un aviso para generar una clave nueva desde el portal.

## Endpoints principales

### Crear una cotización

`POST /cotizaciones`

Cuerpo de ejemplo:

```json
{
  "origen": "CD-SALTA",
  "destino": "CD-ROSARIO",
  "peso_kg": 850,
  "volumen_m3": 3.2,
  "urgencia": "estandar"
}
```

Los valores posibles de `urgencia` son `estandar` (entrega en 72 horas hábiles), `prioritaria` (48 horas hábiles) y `express` (24 horas hábiles). La modalidad `express` solo está disponible entre centros de distribución que estén a menos de 1.200 km de distancia.

La respuesta incluye un `cotizacion_id`, el precio final en pesos argentinos y la fecha estimada de entrega. Una cotización es válida durante **24 horas**; pasado ese tiempo hay que pedir una nueva.

### Reservar un envío

`POST /cotizaciones/{cotizacion_id}/reserva`

Convierte una cotización vigente en un envío. Devuelve un `envio_id` y el código de seguimiento que se le comparte al destinatario. Una cotización solo se puede reservar una vez.

### Consultar una cotización

`GET /cotizaciones/{cotizacion_id}`

Devuelve el estado de la cotización: `vigente`, `reservada` o `vencida`.

## Cómo se calcula el precio

El precio se calcula a partir del **peso facturable**, que es el mayor entre el peso real y el peso volumétrico. El peso volumétrico se obtiene multiplicando el volumen en metros cúbicos por 250. Por ejemplo, un envío de 3,2 m³ tiene un peso volumétrico de 800 kg; si pesa 850 kg, el peso facturable es 850 kg.

Sobre la tarifa base se aplican estos recargos:

- `prioritaria`: 35 % sobre la tarifa base.
- `express`: 80 % sobre la tarifa base.
- Carga peligrosa: recargo fijo por envío, que se informa en la cotización.

Los clientes con contrato anual tienen un descuento que se aplica automáticamente según el identificador de cliente.

## Límites de uso

Cada cliente puede hacer hasta **120 solicitudes por minuto** en producción y 30 por minuto en sandbox. Si se supera el límite, la API responde con el código 429 y el encabezado `Retry-After` indica cuántos segundos hay que esperar.

## Códigos de error

| Código | Error | Significado |
|---|---|---|
| 400 | `DATOS_INVALIDOS` | Falta un campo obligatorio o un valor está fuera de rango |
| 401 | `TOKEN_INVALIDO` | El token no existe o venció |
| 404 | `COTIZACION_INEXISTENTE` | El `cotizacion_id` no existe |
| 409 | `COTIZACION_YA_RESERVADA` | Se intentó reservar dos veces la misma cotización |
| 410 | `COTIZACION_VENCIDA` | Pasaron más de 24 horas desde que se creó la cotización |
| 422 | `RUTA_NO_DISPONIBLE` | No hay servicio entre el origen y el destino con la urgencia pedida |
| 429 | `LIMITE_EXCEDIDO` | Se superó el límite de solicitudes por minuto |

## Soporte

Las consultas técnicas sobre la API se hacen por correo a la mesa de integraciones, que responde en un plazo de un día hábil. Los problemas que afectan envíos en curso se reportan por teléfono a la línea de operaciones, disponible las 24 horas.
