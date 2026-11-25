"""Transform supervised polygon coordinates across real source-image crops."""
import math


def padded_box(box,size,fraction):
    if not math.isfinite(fraction)or fraction<0:raise ValueError('Crop padding must be finite and nonnegative.')
    width,height=size;x0,y0,x1,y1=box
    if not(0<=x0<x1<=width and 0<=y0<y1<=height):raise ValueError('Invalid source crop.')
    dx=(x1-x0)*fraction;dy=(y1-y0)*fraction
    return [max(0,math.floor(x0-dx)),max(0,math.floor(y0-dy)),
            min(width,math.ceil(x1+dx)),min(height,math.ceil(y1+dy))]


def recrop_polygons(polygons,old_box,new_box):
    x0,y0,x1,y1=old_box;a0,b0,a1,b1=new_box
    if a1<=a0 or b1<=b0:raise ValueError('New crop must have positive area.')
    def point(value,old_origin,old_span,new_origin,new_end):
        # PIL truncates each floating polygon vertex to an integer pixel.
        # Preserve that integer through normalized-coordinate rounding. A
        # quarter-pixel offset prevents tiny float errors crossing its boundary;
        # the crop's far edge remains exact and within the 0..1 contract.
        pixel=int(old_origin+value*old_span)
        stable=float(pixel)if pixel==new_end else pixel+.25
        return round((stable-new_origin)/(new_end-new_origin),6)
    return [[[point(u,x0,x1-x0,a0,a1),point(v,y0,y1-y0,b0,b1)]for u,v in polygon]for polygon in polygons]


def normalized_box(box,size):
    normalized=[round(value*1000/size[i%2])for i,value in enumerate(box)]
    if [round(value*size[i%2]/1000)for i,value in enumerate(normalized)]!=box:
        normalized=[round(value*1000/size[i%2],8)for i,value in enumerate(box)]
    return normalized
