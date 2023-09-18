import numpy as np
np.set_printoptions(threshold=np.inf)
def getFeatures(data):
    p,e,d = data.shape
    data = data.astype(float) / 255.0
    n=e*d

    grayscale = np.zeros((n, p))
    x = np.zeros((n, 2))
    y = np.zeros(n)
    c=0

    for i in range(d):
        for j in range(e):
            c += 1  # Update count
            if i < d - 1:
                y[c - 1] = i + 1  # Classify digits 1-9
            else:
                y[c - 1] = 0  # Classify digit 0
            x[c - 1, 0] = 2 * np.mean(data[:, j, i]) - 1  # Calculate average intensity and normalize between -1 and 1
            grayscale[c - 1, :] = data[:, j, i].flatten()  # Create array of pixel values
    w = int(np.floor(np.sqrt(p)))
    sym = np.zeros((n, 2))
    for i in range(n):
        full = grayscale[i, :]
        for j in range(w // 2 ):
            jsym = w + 1 - j
            idxH = slice((j - 1) * w, j * w)
            idxV = slice(j, (w - 1) * w + j, w)
            idxHsym = slice((jsym - 1) * w, jsym * w)
            idxVsym = slice(jsym, (w - 1) * w + jsym, w)
            H = full[idxH]
            Hsym = full[idxHsym]
            V = full[idxV]
            Vsym = full[idxVsym]

            FH=np.abs(H - Hsym)
            VH=np.abs(V - Vsym)

            if( FH.size != 0):
             sym[i, 0] += np.mean(FH)

            else:
             sym[i,0]+= 0


            if(VH.size != 0):
             sym[i, 1] += np.mean(VH)
            else:
             sym[i, 1] += 0

    totsym = np.mean(sym, axis=1)
    x[:, 1] = -totsym / 2 + 1
    print(x)
    return y , x





