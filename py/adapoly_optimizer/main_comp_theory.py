

def belong(G, W):
    # 检测字符串所有字符是否属于终极符与非终极符或为空
    if W == '':
        return True
    match_chars = G[0]+G[1]
    for w in W:
        for i, c in enumerate(match_chars):
            if w == c:
                break
            if i is len(match_chars)-1:
                return False
    return True
        


if __name__ == '__main__':
    # G=[V,T,S,P]
    g = [['A','B'],['1','2','#'],'S',[('A','1A2'),('A','B'),('B','#')]]
    w = '121A21C'
    if belong(g, w):
        print('String %s belongs grammer %s' % (w, g))
    else:
        print('String %s does not belong grammer %s' % (w, g))
