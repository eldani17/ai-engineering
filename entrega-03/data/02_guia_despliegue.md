# Guía de despliegue de Kairo

> Documento interno de Transportes Altiplano S.A. Equipo de Plataforma. Aplica a todos los servicios de Kairo.

## Ambientes

Kairo tiene tres ambientes:

1. **Desarrollo local**: cada desarrollador levanta los servicios que necesita con `make dev`, que usa Docker Compose y datos de prueba sintéticos.
2. **Staging** (`kairo-staging`): réplica reducida de producción. Se actualiza automáticamente con cada merge a la rama `main`.
3. **Producción** (`kairo-prod`): solo se actualiza mediante un release etiquetado.

El clúster `kairo-sandbox` no forma parte del circuito de despliegue: lo usan los clientes corporativos para probar integraciones y recibe la misma versión que producción cada domingo, cuando se recrea.

## Flujo de un cambio

1. El desarrollador abre un pull request contra `main`. El pipeline de integración continua corre los tests unitarios, los tests de contrato de gRPC y el análisis estático.
2. El pull request necesita **dos aprobaciones**, y al menos una tiene que ser de alguien del equipo dueño del servicio.
3. Al hacer merge, Argo CD despliega automáticamente en staging.
4. En staging corren las pruebas de humo, que tardan unos 12 minutos. Si fallan, el equipo dueño recibe un aviso en el canal `#kairo-deploys`.
5. Para pasar a producción se crea una etiqueta con el formato `servicio/vAÑO.MES.NÚMERO`, por ejemplo `despacho/v2026.03.4`. La etiqueta dispara el despliegue en producción.

## Despliegue progresivo

En producción los servicios se despliegan con la estrategia canary, usando Argo Rollouts:

- Primero recibe la versión nueva el 10 % del tráfico durante 15 minutos.
- Si la tasa de errores no supera el 1 % y la latencia p99 no sube más de un 20 %, se pasa al 50 % durante otros 15 minutos.
- Si las métricas siguen sanas, se completa el despliegue al 100 %.

Si alguna de las métricas se pasa de los umbrales, Argo Rollouts revierte automáticamente a la versión anterior y deja un aviso en `#kairo-deploys`. El servicio `despacho` usa umbrales más estrictos: la tasa de errores no puede superar el 0,5 %.

## Ventanas de congelamiento

No se despliega a producción en estos momentos:

- **Viernes desde las 15:00 hasta el lunes a las 09:00** (hora de Argentina).
- **Del 20 de diciembre al 6 de enero**, por el pico de envíos de fin de año.
- **Durante el cierre contable mensual**, que son los últimos dos días hábiles de cada mes. Este congelamiento aplica solo a `facturador` y `cotizador`.

Para desplegar durante una ventana de congelamiento hace falta una excepción aprobada por la persona que esté de guardia como líder de incidentes y por el responsable del equipo dueño. Las excepciones se piden en el canal `#kairo-excepciones` con el formulario fijado en el canal, y solo se aprueban para correcciones de incidentes SEV1 o SEV2.

## Rollback manual

Si un problema aparece después de completado el despliegue, el rollback se hace desde Argo CD:

1. Entrar a la aplicación del servicio en Argo CD.
2. Elegir "History and rollback" y seleccionar la revisión anterior.
3. Confirmar y avisar en `#kairo-deploys` con el motivo.

El rollback **no revierte migraciones de base de datos**. Por eso las migraciones tienen que ser compatibles hacia atrás: primero se agrega una columna nueva sin quitar la vieja, se despliega el código que usa las dos, y recién en un release posterior se elimina la columna vieja. Esta regla se conoce internamente como "migraciones en dos pasos".

## Secretos y configuración

Los secretos (contraseñas de bases de datos, tokens de proveedores de SMS, certificados) se guardan en Vault y se inyectan en los pods mediante el operador External Secrets. Nunca se guardan secretos en los repositorios `kairo-deploy` ni `kairo-infra`. Si alguien sube un secreto por error, tiene que rotarlo de inmediato aunque haya borrado el commit, y avisar al equipo de Seguridad.

La configuración que no es secreta (niveles de log, flags de funcionalidades, límites) vive en archivos `values-prod.yaml` y `values-staging.yaml` dentro de `kairo-deploy`.

## Feature flags

Las funcionalidades nuevas se activan con feature flags administradas en la herramienta interna Interruptor. Una flag nueva arranca apagada en producción. Se recomienda encenderla primero para un solo centro de distribución, normalmente el de Córdoba, que es el de menor volumen, y después extenderla al resto. Las flags que llevan más de 90 días encendidas para todos tienen que eliminarse del código.
