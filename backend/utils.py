def spectral_to_temp(s):
    mapping = {
        "O":30000,
        "B":20000,
        "A":8500,
        "F":6500,
        "G":5500,
        "K":4500,
        "M":3200
    }
    return mapping.get(s.upper(), 5500)


def compute_stellar_compatibility(eq_temp):
    ideal = 288
    return max(0, 1 - abs(eq_temp - ideal)/500)
