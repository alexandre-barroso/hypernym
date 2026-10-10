import numpy as np


def parse_vector(line, dimensions):
    pieces=line.rstrip('\r\n').rsplit(' ',dimensions)
    if len(pieces)!=dimensions+1 or not pieces[0]:
        raise ValueError('Malformed vector row')
    vector=np.asarray([float(x) for x in pieces[1:]],dtype=np.float64)
    if not np.isfinite(vector).all():raise ValueError('Nonfinite vector')
    return pieces[0],vector

def merge_moments(count,mean,m2,block):
    """Parallel central-moment merge; avoids subtracting two large raw sums."""
    bcount=len(block);bmean=block.mean(axis=0);center=block-bmean
    bm2=center.T@center
    if not count:return bcount,bmean,bm2
    delta=bmean-mean;total=count+bcount
    return total,mean+delta*(bcount/total),m2+bm2+np.outer(delta,delta)*(count*bcount/total)

def parse_header(raw):
    parts=raw.decode('utf-8').strip().split()
    if len(parts)!=2 or any(not x.isascii() or not x.isdecimal() for x in parts):
        raise ValueError('Malformed fastText header')
    rows,dimension=map(int,parts)
    if rows<2 or dimension<2:raise ValueError('Insufficient vector shape')
    return rows,dimension

def parse_row(raw,dimension):
    # The documented format has one word followed by space-separated numbers.
    # Splitting from the right also preserves a literal Unicode word identity.
    parts=raw.decode('utf-8').rstrip('\r\n').rsplit(' ',dimension)
    if len(parts)!=dimension+1 or not parts[0] or any(not x for x in parts[1:]):
        raise ValueError('Malformed static vector row')
    if any(x in parts[0] for x in ('\n','\r','\t','\x00')):
        raise ValueError('Unexpected control character in lexical token')
    vector=np.array([float(x) for x in parts[1:]],dtype=np.float64)
    if not np.isfinite(vector).all():raise ValueError('Nonfinite vector row')
    return parts[0],vector

def eigendecomposition(covariance):
    values,components=np.linalg.eigh(covariance)
    values=values[::-1];components=components[:,::-1]
    for j in range(len(values)):
        i=int(np.argmax(np.abs(components[:,j])))
        if components[i,j]<0:components[:,j]*=-1
    scale=max(float(np.linalg.norm(covariance,ord=2)),1e-300)
    residual=float(np.linalg.norm(covariance@components-components*values,ord=2)/scale)
    orthogonal=float(np.linalg.norm(components.T@components-np.eye(len(values)),ord=2))
    if values[-1]<-1e-12*scale or residual>1e-10 or orthogonal>1e-10:
        raise ValueError('PCA numerical validation failed')
    tolerance=64*np.finfo(np.float64).eps*len(values)*scale
    gaps=values[:-1]-values[1:]
    return values,components,dict(relative_residual=residual,orthogonality_error=orthogonal,
        gap_screen_tolerance=tolerance,minimum_adjacent_gap=float(gaps.min()),
        numerically_ambiguous_prefix_boundaries=[int(i+1) for i in np.flatnonzero(gaps<=tolerance)],
        gap_screen_scope='Floating-point ambiguity screen, not proof of exact population eigenvalue separation')
