#!/usr/bin/env python3
"""
Script principal para probar el agente conversacional LLM con herramientas de transparencia.

Este script permite probar diferentes tipos de consultas y verificar que el sistema funcione correctamente después de la refactorización.
"""

import asyncio
import sys
from typing import Dict, List

from app.llm import run, run_stream

# Preguntas de prueba por categoría
PREGUNTAS_PRUEBA: Dict[str, List[str]] = {
    "votaciones": [
        "Que asuntos se trataron en las votaciones del mes de octubre del 2022",
        "¿Qué rango de fechas tienes disponible para votaciones?",
    ],
    "asistencias": [
        "Dame las asistencias de octubre del 2022",
        "¿Qué rango de fechas tienes para asistencias parlamentarias?",
    ],
    "contrataciones": [
        "Dame la lista de empresas con más de 100000000 de soles en contratos con el Estado",
        "Dame la lista de empresas que iniciaron sus actividades en el 2022 y dentro de sus 3 primeros meses de actividad tuvieron contratos por más de 300000 soles",
    ],
    "busqueda_web": [
        "¿Quién es el congresista Alejandro Muñante?",
        "Busca información sobre transparencia gubernamental en Perú",
    ],
    "fallback": [
        "¿Cuál es la capital de Francia?",
        "¿Cómo está el clima hoy?",
        "Hola, ¿cómo estás?",
    ],
}


async def probar_pregunta(pregunta: str, usar_streaming: bool = False) -> None:
    """
    Prueba una pregunta individual con el LLM.

    Args:
        pregunta: La pregunta a probar
        usar_streaming: Si usar streaming o respuesta completa
    """
    print(f"\n{'='*80}")
    print(f"PREGUNTA: {pregunta}")
    print(f"{'='*80}")

    try:
        if usar_streaming:
            print("RESPUESTA (streaming):")
            print("-" * 40)

            async for chunk in run_stream(pregunta):
                if chunk["is_complete"]:
                    print(f"\n\n[COMPLETADO - Nodo: {chunk['node']}]")
                    print(f"Thread ID: {chunk['thread_id']}")
                else:
                    print(chunk["token"], end="", flush=True)
        else:
            resultado = await run(pregunta)
            print("RESPUESTA:")
            print("-" * 40)
            print(resultado["response"])
            print(f"\nThread ID: {resultado['thread_id']}")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()


async def probar_categoria(
    categoria: str, preguntas: List[str], usar_streaming: bool = False
) -> None:
    """
    Prueba todas las preguntas de una categoría.

    Args:
        categoria: Nombre de la categoría
        preguntas: Lista de preguntas a probar
        usar_streaming: Si usar streaming o respuesta completa
    """
    print(f"\n🔍 PROBANDO CATEGORÍA: {categoria.upper()}")

    for i, pregunta in enumerate(preguntas, 1):
        print(f"\n[{i}/{len(preguntas)}]")
        await probar_pregunta(pregunta, usar_streaming)

        # Pausa entre preguntas para no saturar
        if i < len(preguntas):
            await asyncio.sleep(1)


async def menu_interactivo():
    """Menú interactivo para probar el sistema."""
    while True:
        print(f"\n{'='*60}")
        print("🤖 MENÚ DE PRUEBAS - AGENTE LLM TRANSPARENCIA")
        print(f"{'='*60}")
        print("1. Probar todas las categorías")
        print("2. Probar categoría específica")
        print("3. Pregunta personalizada")
        print("4. Pregunta personalizada (streaming)")
        print("5. Salir")
        print("-" * 60)

        try:
            opcion = input("Selecciona una opción (1-5): ").strip()

            if opcion == "1":
                usar_streaming = input("¿Usar streaming? (s/N): ").strip().lower() == "s"
                for categoria, preguntas in PREGUNTAS_PRUEBA.items():
                    await probar_categoria(categoria, preguntas, usar_streaming)

            elif opcion == "2":
                print("\nCategorías disponibles:")
                for i, categoria in enumerate(PREGUNTAS_PRUEBA.keys(), 1):
                    print(f"{i}. {categoria}")

                try:
                    cat_num = int(input("Selecciona categoría (número): ").strip())
                    categorias = list(PREGUNTAS_PRUEBA.keys())
                    if 1 <= cat_num <= len(categorias):
                        categoria = categorias[cat_num - 1]
                        usar_streaming = input("¿Usar streaming? (s/N): ").strip().lower() == "s"
                        await probar_categoria(
                            categoria, PREGUNTAS_PRUEBA[categoria], usar_streaming
                        )
                    else:
                        print("❌ Número de categoría inválido")
                except ValueError:
                    print("❌ Por favor ingresa un número válido")

            elif opcion == "3":
                pregunta = input("Ingresa tu pregunta: ").strip()
                if pregunta:
                    await probar_pregunta(pregunta, usar_streaming=False)
                else:
                    print("❌ Pregunta vacía")

            elif opcion == "4":
                pregunta = input("Ingresa tu pregunta: ").strip()
                if pregunta:
                    await probar_pregunta(pregunta, usar_streaming=True)
                else:
                    print("❌ Pregunta vacía")

            elif opcion == "5":
                print("👋 ¡Hasta luego!")
                break

            else:
                print("❌ Opción inválida. Por favor selecciona 1-5.")

        except KeyboardInterrupt:
            print("\n\n👋 Interrumpido por el usuario. ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error inesperado: {e}")


async def modo_rapido():
    """Modo rápido: prueba una pregunta de cada categoría."""
    print("🚀 MODO RÁPIDO - Una pregunta por categoría")

    for categoria, preguntas in PREGUNTAS_PRUEBA.items():
        print(f"\n🔍 Probando {categoria}...")
        await probar_pregunta(preguntas[0], usar_streaming=False)
        await asyncio.sleep(0.5)  # Pausa breve


async def main():
    """Función principal."""
    print("🤖 Agente Conversacional LLM - Sistema de Transparencia Gubernamental")
    print("=" * 80)

    if len(sys.argv) > 1:
        if sys.argv[1] == "--rapido":
            await modo_rapido()
        elif sys.argv[1] == "--pregunta":
            if len(sys.argv) > 2:
                pregunta = " ".join(sys.argv[2:])
                await probar_pregunta(pregunta)
            else:
                print("❌ Uso: python main.py --pregunta 'tu pregunta aquí'")
        elif sys.argv[1] == "--streaming":
            if len(sys.argv) > 2:
                pregunta = " ".join(sys.argv[2:])
                await probar_pregunta(pregunta, usar_streaming=True)
            else:
                print("❌ Uso: python main.py --streaming 'tu pregunta aquí'")
        else:
            print("❌ Opciones disponibles:")
            print("  python main.py                    # Menú interactivo")
            print("  python main.py --rapido           # Prueba rápida")
            print("  python main.py --pregunta 'texto' # Pregunta directa")
            print("  python main.py --streaming 'texto'# Pregunta con streaming")
    else:
        await menu_interactivo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 ¡Hasta luego!")
    except Exception as e:
        print(f"❌ Error fatal: {e}")
        sys.exit(1)
