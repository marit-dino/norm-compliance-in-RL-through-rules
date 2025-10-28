from shield.rule_classes import Fact

turn_90_2_ghosts = ( # tuple of (action-relations, feature-relations)
        {0 : 2,
         2 : 1,
         1 : 3,
         3 : 0
         }, {
        #closest-capsule-dir
        2 : 4,
        3 : 5,
        4 : 3,
        5 : 2,
            # closest food direction
        9 : 11,
        10 : 12,
        11 : 10,
        12 : 9,
         # ghost-0-angle
        15 : 17,
        16 : 18,
        17 : 19,
        18 : 20,
        19 : 21,
        20 : 22,
        21 : 15,
        22 : 16,
        # ghost-0-dir
        23 : 25,
        24 : 26,
        25 : 24,
        26 : 23,
        # ghost-0-heading
        29 : 31,
        30 : 32,
        31 : 30,
        32 : 29,
        # ghost-1-angle
        39 : 41,
        40 : 42,
        41 : 43,
        42 : 44,
        43 : 45,
        44 : 46,
        45 : 39,
        46 : 40,
        # ghost-1-dir
        47 : 49,
        48 : 50,
        49 : 48,
        50 : 47,
        # ghost-1-heading
        53 : 55,
        54 : 56,
        55 : 54,
        56 : 53,
        # poss-dir
        62 : 64,
        63 : 65,
        64 : 63,
        65 : 62,

        # -of-ghosts-1-step-away
        68: 70,
        69: 71,
        70: 69,
        71: 68,
        # -of-scared-ghosts-1-step-away
        73: 75,
        74: 76,
        75: 74,
        76: 73
        }
    )

turn_90_4_ghosts = (  # tuple of (action-relations, feature-relations)
    {0: 2,
     2: 1,
     1: 3,
     3: 0
     }, {
        # closest-capsule-dir
        2: 4,
        3: 5,
        4: 3,
        5: 2,
        # closest food direction
        9: 11,
        10: 12,
        11: 10,
        12: 9,
        # ghost-0-angle
        15: 17,
        16: 18,
        17: 19,
        18: 20,
        19: 21,
        20: 22,
        21: 15,
        22: 16,
        # ghost-0-dir
        23: 25,
        24: 26,
        25: 24,
        26: 23,
        # ghost-0-heading
        29: 31,
        30: 32,
        31: 30,
        32: 29,
        # ghost-1-angle
        39: 41,
        40: 42,
        41: 43,
        42: 44,
        43: 45,
        44: 46,
        45: 39,
        46: 40,
        # ghost-1-dir
        47: 49,
        48: 50,
        49: 48,
        50: 47,
        # ghost-1-heading
        53: 55,
        54: 56,
        55: 54,
        56: 53,
        #ghost-2-angle
        63 : 65,
        64 : 66,
        65 : 67,
        66 : 68,
        67 : 69,
        68 : 70,
        69 : 63,
        70 : 64,
        # ghost-2-dir
        71 : 73,
        72 : 74,
        73 : 72,
        74 : 71,
        # ghost-2-heading
        77 : 79,
        78 : 80,
        79 : 78,
        80 : 77,
        # ghost-3-angle
        87 : 89,
        88 : 90,
        89 : 91,
        90 : 92,
        91 : 93,
        92 : 94,
        93 : 87,
        94 : 88,
        # ghost-3-dir
        95 : 97,
        96 : 98,
        97 : 96,
        98 : 95,
        # ghost-3-heading
        101 : 103,
        102 : 104,
        103 : 102,
        104 : 101,
        # poss-dir
        110: 112,
        111: 113,
        112: 111,
        113: 110,
        #-of-ghosts-1-step-away
        116: 118,
        117: 119,
        118: 117,
        119: 116,
        #-of-scared-ghosts-1-step-away
        121 : 123,
        122 : 124,
        123 : 122,
        124 : 121
    }
)

# add remove features somehow for coordinates
remove_feat_2_ghosts = [67,68,36,37,60,61]
remove_feat_4_ghosts = [36,37,60,61,84,85,108,109,115,116]

# action -> feature safeguards when enforcing actions
safe_guard_2_ghosts = {
    0 : Fact(62,1),
    1 : Fact(63,1),
    2 : Fact(64,1),
    3 : Fact(65,1)
}
safe_guard_4_ghosts = {
    0 : Fact(110,1),
    1 : Fact(111,1),
    2 : Fact(112,1),
    3 : Fact(113,1)
}