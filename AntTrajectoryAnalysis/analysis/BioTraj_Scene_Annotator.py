#!/usr/bin/env python3
"""BioTraj Scene Annotator v1.0 — polygons, circles, polylines, labels and artifact masks."""
import argparse,json,math,tkinter as tk
from pathlib import Path
from tkinter import ttk,filedialog,messagebox,simpledialog
from PIL import Image,ImageTk,ImageDraw,ImageFont
VERSION='1.0'
CATS=['surface','surface_transition','biological','artifact','physical_constraint','observation_boundary','other']
COL={'surface':'#2E86DE','surface_transition':'#F39C12','biological':'#27AE60','artifact':'#E74C3C','physical_constraint':'#8E44AD','observation_boundary':'#7F8C8D','other':'#16A085'}
class App:
 def __init__(self,r,path=None):
  self.r=r;r.title('BioTraj Scene Annotator v'+VERSION);r.geometry('1450x900')
  self.path=None;self.im=None;self.tk=None;self.scale=1;self.ox=self.oy=10;self.anns=[];self.nid=1;self.pts=[];self.cc=None;self.sel=None
  self.tool=tk.StringVar(value='polygon');self.cat=tk.StringVar(value='surface');self.lab=tk.StringVar(value='region');self.exc=tk.BooleanVar();self.status=tk.StringVar(value='Open an image.')
  self.ui();self.keys()
  if path:self.load_image(path)
 def ui(self):
  t=ttk.Frame(self.r,padding=5);t.pack(fill='x')
  for txt,cmd in [('Open image',self.open_image),('Load JSON',self.load_json),('Save JSON',self.save_json),('Export PNG',self.export_png)]:ttk.Button(t,text=txt,command=cmd).pack(side='left',padx=2)
  ttk.Label(t,text='  Tool:').pack(side='left')
  for txt,v in [('Polygon','polygon'),('Circle','circle'),('Polyline','polyline')]:ttk.Radiobutton(t,text=txt,variable=self.tool,value=v,command=self.cancel).pack(side='left')
  ttk.Label(t,text='  Category:').pack(side='left');ttk.Combobox(t,textvariable=self.cat,values=CATS,width=19).pack(side='left')
  ttk.Label(t,text='  Label:').pack(side='left');ttk.Entry(t,textvariable=self.lab,width=18).pack(side='left')
  ttk.Checkbutton(t,text='Exclude / artifact mask',variable=self.exc).pack(side='left',padx=8)
  p=ttk.Panedwindow(self.r,orient='horizontal');p.pack(fill='both',expand=True);cf=ttk.Frame(p);sf=ttk.Frame(p,padding=6);p.add(cf,weight=5);p.add(sf,weight=1)
  self.c=tk.Canvas(cf,bg='#222',highlightthickness=0);self.c.pack(fill='both',expand=True)
  self.c.bind('<Button-1>',self.click);self.c.bind('<Button-3>',lambda e:self.finish());self.c.bind('<Double-Button-1>',lambda e:self.finish());self.c.bind('<Configure>',lambda e:self.redraw())
  ttk.Label(sf,text='Annotations',font=('',12,'bold')).pack(anchor='w');self.lb=tk.Listbox(sf,exportselection=False);self.lb.pack(fill='both',expand=True,pady=5);self.lb.bind('<<ListboxSelect>>',self.choose)
  a=ttk.Frame(sf);a.pack(fill='x');ttk.Button(a,text='Rename',command=self.rename).pack(side='left',fill='x',expand=True);ttk.Button(a,text='Toggle exclude',command=self.toggle).pack(side='left',fill='x',expand=True)
  b=ttk.Frame(sf);b.pack(fill='x');ttk.Button(b,text='Delete',command=self.delete).pack(side='left',fill='x',expand=True);ttk.Button(b,text='Undo last',command=self.undo).pack(side='left',fill='x',expand=True)
  help='Polygon/polyline: left-click vertices; right-click or Enter to finish.\nCircle: click centre, then edge.\n\nUse artifact + Exclude for wind-blown leaves, moving vegetation, reflections, etc.\n\nEnter finish | Esc cancel | Delete selected | Ctrl+S save\n\nAll coordinates are original-image pixels.'
  ttk.Label(sf,text=help,justify='left',wraplength=300).pack(anchor='w',pady=8);ttk.Label(self.r,textvariable=self.status,relief='sunken',anchor='w').pack(fill='x')
 def keys(self):
  self.r.bind('<Return>',lambda e:self.finish());self.r.bind('<Escape>',lambda e:self.cancel());self.r.bind('<Delete>',lambda e:self.delete());self.r.bind('<Control-s>',lambda e:self.save_json())
 def open_image(self):
  p=filedialog.askopenfilename(filetypes=[('Images','*.jpg *.jpeg *.png *.tif *.tiff *.bmp'),('All','*.*')]);
  if p:self.load_image(p)
 def load_image(self,p):
  try:self.path=Path(p);self.im=Image.open(p).convert('RGB');self.anns=[];self.nid=1;self.sel=None;self.status.set(f'{self.path.name}: {self.im.width} x {self.im.height} px');self.r.after(100,self.redraw);self.refresh()
  except Exception as e:messagebox.showerror('Open image',str(e))
 def redraw(self):
  if not self.im:return
  w=max(100,self.c.winfo_width()-20);h=max(100,self.c.winfo_height()-20);self.scale=min(w/self.im.width,h/self.im.height,1.0);dw=int(self.im.width*self.scale);dh=int(self.im.height*self.scale)
  self.tk=ImageTk.PhotoImage(self.im.resize((dw,dh),Image.Resampling.LANCZOS));self.c.delete('all');self.c.create_image(10,10,anchor='nw',image=self.tk)
  for a in self.anns:self.draw(a)
  col=COL.get(self.cat.get(),'white');xy=[self.xy(p) for p in self.pts]
  for x,y in xy:self.c.create_oval(x-3,y-3,x+3,y+3,fill=col,outline='')
  if len(xy)>1:self.c.create_line(*[v for q in xy for v in q],fill=col,width=2)
  if self.cc:
   x,y=self.xy(self.cc);self.c.create_oval(x-4,y-4,x+4,y+4,fill=col,outline='white')
 def ip(self,e):
  if not self.im:return None
  x=(e.x-10)/self.scale;y=(e.y-10)/self.scale
  return [float(x),float(y)] if 0<=x<self.im.width and 0<=y<self.im.height else None
 def xy(self,p):return 10+p[0]*self.scale,10+p[1]*self.scale
 def click(self,e):
  p=self.ip(e)
  if p is None:return
  if self.tool.get() in ('polygon','polyline'):self.pts.append(p);self.redraw()
  else:
   if self.cc is None:self.cc=p;self.status.set('Circle centre set; click its edge.');self.redraw()
   else:
    r=math.hypot(p[0]-self.cc[0],p[1]-self.cc[1]);self.add({'shape':'circle','center':self.cc,'radius':r});self.cc=None
 def finish(self):
  s=self.tool.get()
  if s=='polygon' and len(self.pts)>=3:self.add({'shape':s,'points':self.pts[:]});self.pts=[]
  elif s=='polyline' and len(self.pts)>=2:self.add({'shape':s,'points':self.pts[:]});self.pts=[]
  elif self.pts:self.pts=[];self.status.set('Incomplete annotation cancelled.');self.redraw()
 def cancel(self):self.pts=[];self.cc=None;self.redraw()
 def add(self,g):
  a={'id':self.nid,'label':self.lab.get().strip() or f'annotation_{self.nid}','category':self.cat.get().strip() or 'other','exclude_from_tracking_analysis':bool(self.exc.get()),**g};self.nid+=1;self.anns.append(a);self.sel=a['id'];self.refresh();self.redraw()
 def color(self,a):return COL['artifact'] if a.get('exclude_from_tracking_analysis') else COL.get(a.get('category'),'#16A085')
 def draw(self,a):
  col=self.color(a);wid=4 if a['id']==self.sel else 2;dash=(7,4) if a.get('exclude_from_tracking_analysis') else None
  if a['shape'] in ('polygon','polyline'):
   q=[self.xy(p) for p in a['points']];flat=[v for z in q for v in z]
   if a['shape']=='polygon':self.c.create_polygon(*flat,fill='',outline=col,width=wid,dash=dash)
   else:self.c.create_line(*flat,fill=col,width=wid,dash=dash)
   if q:self.c.create_text(q[0][0]+4,q[0][1]-4,anchor='sw',text=a['label'],fill=col,font=('',10,'bold'))
  else:
   x,y=self.xy(a['center']);rr=a['radius']*self.scale;self.c.create_oval(x-rr,y-rr,x+rr,y+rr,outline=col,width=wid,dash=dash);self.c.create_text(x,y-rr-3,anchor='s',text=a['label'],fill=col,font=('',10,'bold'))
 def refresh(self):
  self.lb.delete(0,'end')
  for a in self.anns:self.lb.insert('end',f"{a['id']:03d} | {a['category']} | {a['shape']} | {a['label']}"+(' [EXCLUDE]' if a.get('exclude_from_tracking_analysis') else ''))
 def choose(self,e=None):
  s=self.lb.curselection()
  if s:self.sel=self.anns[s[0]]['id'];self.redraw()
 def selected(self):return next((a for a in self.anns if a['id']==self.sel),None)
 def rename(self):
  a=self.selected()
  if a:
   s=simpledialog.askstring('Rename','Label:',initialvalue=a['label'])
   if s and s.strip():a['label']=s.strip();self.refresh();self.redraw()
 def toggle(self):
  a=self.selected()
  if a:a['exclude_from_tracking_analysis']=not a.get('exclude_from_tracking_analysis',False);self.refresh();self.redraw()
 def delete(self):
  if self.sel is not None:self.anns=[a for a in self.anns if a['id']!=self.sel];self.sel=None;self.refresh();self.redraw()
 def undo(self):
  if self.pts or self.cc:self.cancel()
  elif self.anns:self.anns.pop();self.refresh();self.redraw()
 def obj(self):return {'format':'BioTraj_scene_annotations','version':VERSION,'image':self.path.name if self.path else None,'image_width':self.im.width,'image_height':self.im.height,'coordinate_system':'image pixels; origin top-left; x right; y down','annotations':self.anns}
 def save_json(self):
  if not self.im:return
  d=self.path.with_name(self.path.stem+'_scene_annotations.json');p=filedialog.asksaveasfilename(initialdir=d.parent,initialfile=d.name,defaultextension='.json',filetypes=[('JSON','*.json')])
  if p:Path(p).write_text(json.dumps(self.obj(),indent=2));self.status.set('Saved '+p)
 def load_json(self):
  p=filedialog.askopenfilename(filetypes=[('JSON','*.json')])
  if not p:return
  try:
   o=json.loads(Path(p).read_text())
   if not self.im:
    q=Path(p).parent/o.get('image','')
    if q.exists():self.load_image(q)
    else:messagebox.showerror('Load JSON','Open the corresponding image first.');return
   if (o.get('image_width'),o.get('image_height'))!=(self.im.width,self.im.height) and not messagebox.askyesno('Size differs','JSON and current image dimensions differ. Load anyway?'):return
   self.anns=o.get('annotations',[]);self.nid=max([a.get('id',0) for a in self.anns]+[0])+1;self.refresh();self.redraw()
  except Exception as e:messagebox.showerror('Load JSON',str(e))
 def export_png(self):
  if not self.im:return
  dflt=self.path.with_name(self.path.stem+'_scene_annotated.png');p=filedialog.asksaveasfilename(initialdir=dflt.parent,initialfile=dflt.name,defaultextension='.png',filetypes=[('PNG','*.png')])
  if not p:return
  out=self.im.copy();d=ImageDraw.Draw(out,'RGBA')
  try:font=ImageFont.truetype('DejaVuSans.ttf',max(16,self.im.width//90))
  except:font=ImageFont.load_default()
  def rgb(h):h=h.lstrip('#');return tuple(int(h[i:i+2],16) for i in (0,2,4))
  for a in self.anns:
   c=rgb(self.color(a));w=max(3,self.im.width//600)
   if a['shape'] in ('polygon','polyline'):
    pts=[tuple(q) for q in a['points']];d.line(pts+([pts[0]] if a['shape']=='polygon' else []),fill=c+(255,),width=w);x,y=pts[0]
   else:x,y=a['center'];rr=a['radius'];d.ellipse((x-rr,y-rr,x+rr,y+rr),outline=c+(255,),width=w);y-=rr
   txt=a['label']+(' [EXCLUDE]' if a.get('exclude_from_tracking_analysis') else '');bb=d.textbbox((x,y),txt,font=font);d.rectangle((bb[0]-2,bb[1]-2,bb[2]+2,bb[3]+2),fill=(255,255,255,190));d.text((x,y),txt,fill=c+(255,),font=font)
  out.save(p);self.status.set('Exported '+p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('image',nargs='?');a=ap.parse_args();r=tk.Tk();App(r,a.image);r.mainloop()
if __name__=='__main__':main()
