"""
SAT => 3SAT
StudentNum: R1202208
F = (¬q1 v ¬q2)∧(q1 v q2 v ¬q3 v q4 v ¬q5)∧(¬q2)
"""


def F(q1, q2, q3, q4, q5):
    C1 = not q1 or not q2
    C2 = q1 or q2 or not q3 or q4 or not q5
    C3 = not q2
    return C1 and C2 and C3


def F_3SAT(q1, q2, q3, q4, q5, y1, y21, y22, y31, y32):
    """
        C1 = ¬q1 v ¬q2
        =>
        C1 = (¬q1 v ¬q2 v y1)
            ∧(¬q1 v ¬q2 v ¬y1)
    """
    C1_3SAT = (not q1 or not q2 or y1) and (not q1 or not q2 or not y1)

    """
        C2 = q1 v q2 v ¬q3 v q4 v ¬q5
        =>
        C2 = (q1 v q2 v y21)
            ∧(¬y21 v ¬q3 v y22)
            ∧(¬y22 v q4 v ¬q5)
    """
    C2_3SAT = (q1 or q2 or y21) and (
        not y21 or not q3 or y22) and (not y22 or q4 or not q5)

    """
        C3 = ¬q2
        =>
        C3 = (¬q2 v y31 v y32)
            ∧(¬q2 v ¬y31 v y32)
            ∧(¬q2 v y31 v ¬y32)
            ∧(¬q2 v ¬y31 v ¬y32)
    """
    C3_3SAT = (not q2 or y31 or y32) and (not q2 or not y31 or y32) and (
        not q2 or y31 or not y32) and (not q2 or not y31 or not y32)

    return C1_3SAT and C2_3SAT and C3_3SAT


"""
    Find some solutions for F_3SAT.
    q1:true, q2:false, q3:false, q4:true, q5:false, y1:true, y21:true, y22:true, y31:true, y32: false
    q1:false, q2:false, q3:false, q4:true, q5:true, y1:false, y21:true, y22:true, y31:true, y32: true
    q1:true, q2:false, q3:true, q4:false, q5:false, y1:false, y21:false, y22:false, y31:false, y32: false
"""
print(F_3SAT(True, False, False, True, False, True, True, True, True, False))
print(F_3SAT(False, False, False, True, True, False, True, True, True, True))
print(F_3SAT(True, False, True, False, False, False, False, False, False, False))

"""
    Verify if the solutions are to F.
"""
print(F(True, False, False, True, False))
print(F(False, False, False, True, True))
print(F(True, False, True, False, False))
