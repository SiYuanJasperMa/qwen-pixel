"""Integer ROI geometry in Qwen's usual0-1000 convention, with native RGB output."""
from .clean_edit_protocol import TOOLS,SYSTEM as FRACTION_SYSTEM,USER_GUIDANCE as FRACTION_GUIDANCE
from .clean_edit_protocol import initial as fraction_initial,program as fraction_program

SYSTEM=FRACTION_SYSTEM.replace(
    'Mask polygon points are (u,v) fractions in 0-1 WITHIN the selected region; convert using the region\'s source offset and size. These fractions are different from the full-image bbox_2d units.',
    'Mask polygon points are integer (u,v) coordinates in 0-1000 WITHIN the selected region. Convert to source pixels using x0+round(u*region_width/1000) and y0+round(v*region_height/1000). bbox_2d refers to the FULL image, while polygon points refer to the selected CROP; these are different coordinate frames.')
USER_GUIDANCE='\nLocate the entire visible target first. Read its actual region before editing. bbox_2d uses 0-1000 full-image coordinates; Python mask polygon points are integers0-1000 within the selected crop.'


def initial(source,instruction):
    messages=fraction_initial(source,instruction);messages[0]['content']=SYSTEM
    for block in messages[1]['content']:
        if block.get('type')=='text':block['text']=block['text'].replace(FRACTION_GUIDANCE,USER_GUIDANCE)
    return messages


def integer_polygons(polygons,box):
    x0,y0,x1,y1=box;width,height=x1-x0,y1-y0
    if width<=0 or height<=0 or max(width,height)>1000:
        raise ValueError('Exact integer-coordinate conversion requires positive native crops no larger than1000 pixels.')
    mapped=[]
    for polygon in polygons:
        points=[]
        for u,v in polygon:
            x,y=int(x0+u*width),int(y0+v*height)
            a,b=round((x-x0)*1000/width),round((y-y0)*1000/height)
            if not(0<=a<=1000 and 0<=b<=1000):raise ValueError('Polygon lies outside its source crop.')
            if x0+round(a*width/1000)!=x or y0+round(b*height/1000)!=y:
                raise ValueError('Integer ROI conversion changed native raster coordinates.')
            points.append([a,b])
        mapped.append(points)
    return mapped


def program(polygons,operation,color='red'):
    code=fraction_program(polygons,operation,color)
    old='(x0+u*region_width, y0+v*region_height)'
    if old not in code:raise ValueError('Original program conversion changed.')
    return code.replace(old,'(x0+round(u*region_width/1000.0), y0+round(v*region_height/1000.0))')
