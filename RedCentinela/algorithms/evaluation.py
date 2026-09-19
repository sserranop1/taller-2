import math

from world.game_state import GameState


def base_evaluation_function(state: GameState) -> float:
    """
    Retorna la evaluación base entregada para desarrollar el punto 4.

    Esta función no forma parte del código que debe modificar el estudiante y
    permite probar Minimax antes de desarrollar la heurística del punto 5.
    """
    if state.is_win():
        return 1000.0
    if state.is_lose():
        return -1000.0
    return float(state.get_score())


def evaluation_function(state: GameState) -> float:
    """
    Evalúa un estado desde la perspectiva del defensor MAX.

    Conserva las utilidades terminales de la evaluación base y valora los
    estados de corte combinando el puntaje acumulado, las terminales
    pendientes, la distancia al objetivo, la distancia al intruso, el riesgo
    inmediato de captura y la movilidad disponible.
    """
    if state.is_win() or state.is_lose():
        return base_evaluation_function(state)

    layout = state.layout
    defender = state.defender_position
    intruder = state.intruder_position
    pending = state.pending_terminals

    value = float(state.get_score())
    value -= 100.0 * len(pending)

    if pending:
        distances = [
            layout.distance(defender, terminal)
            for terminal in pending
        ]

        reachable = [
            distance
            for distance in distances
            if math.isfinite(distance)
        ]

        if reachable:
            value -= 10.0 * min(reachable)
        else:
            value -= 500.0

    threat = layout.distance(intruder, defender)

    if math.isfinite(threat):
        value += 5.0 * min(threat, 6.0)

        if threat <= 1.0:
            value -= 200.0
    else:
        value += 30.0

    value += 3.0 * len(state.get_legal_actions(0))

    return max(-999.0, min(999.0, value))
