"""Desktop capture/replay UI. All Tk updates occur on the main thread."""
import json
from pathlib import Path
import queue
import secrets
import threading
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Ellipse
from .receiver import serve
from .pipeline import process
from .dataset import import_phone
from .evaluate import evaluate

ROOT=Path(__file__).resolve().parents[1]


class Dashboard(tk.Tk):
    def __init__(self):
        super().__init__();self.title('OnePlus 12R • MSCKF + IMU + Features + ArUco');self.geometry('1150x850')
        self.messages=queue.Queue();self.receiver_stop=threading.Event();self.rows=[];self.markers={};self.worker_active=False
        self.dataset=tk.StringVar();self.config=tk.StringVar(value=str(ROOT/'config'/'phone.example.json'));self.prior=tk.StringVar()
        self.token=tk.StringVar(value=secrets.token_hex(4));self.port=tk.StringVar(value='8765');self.status=tk.StringVar(value='Ready. Calibrate before processing real phone data.')
        self.baseline=tk.BooleanVar();self.smooth=tk.BooleanVar(value=True)
        top=ttk.Frame(self,padding=8);top.pack(fill='x')
        ttk.Label(top,text='OnePlus visual–inertial SLAM',font=('Segoe UI',18,'bold')).pack(anchor='w')
        for label,var,kind in [('Recording folder',self.dataset,'directory'),('Calibration JSON',self.config,'file'),('Prior map (optional)',self.prior,'file')]:
            row=ttk.Frame(top);row.pack(fill='x',pady=3);ttk.Label(row,text=label,width=23).pack(side='left')
            ttk.Entry(row,textvariable=var).pack(side='left',fill='x',expand=True)
            ttk.Button(row,text='Browse',command=lambda v=var,k=kind:self.choose(v,k)).pack(side='left')
        capture=ttk.LabelFrame(top,text='Phone connection',padding=5);capture.pack(fill='x',pady=5)
        ttk.Label(capture,text='Port').pack(side='left');ttk.Entry(capture,textvariable=self.port,width=7).pack(side='left',padx=5)
        ttk.Label(capture,text='Pairing token').pack(side='left');ttk.Entry(capture,textvariable=self.token,width=18).pack(side='left',padx=5)
        self.listen_button=ttk.Button(capture,text='Start receiver',command=self.listen);self.listen_button.pack(side='left',padx=5)
        ttk.Button(capture,text='Import phone file',command=self.import_capture).pack(side='left')
        controls=ttk.Frame(top);controls.pack(fill='x')
        ttk.Checkbutton(controls,text='Baseline: no ArUco correction',variable=self.baseline).pack(side='left')
        ttk.Checkbutton(controls,text='Optimize map',variable=self.smooth).pack(side='left',padx=8)
        self.process_button=ttk.Button(controls,text='Process dataset',command=self.process);self.process_button.pack(side='left',padx=8)
        ttk.Button(controls,text='Load results',command=self.load_results).pack(side='left')
        ttk.Button(controls,text='Evaluate with ground truth',command=self.evaluate).pack(side='left',padx=8)
        ttk.Label(top,textvariable=self.status,wraplength=1100).pack(fill='x',pady=5)
        self.figure=Figure(figsize=(10,4.6));self.axes=self.figure.add_subplot(121,projection='3d');self.confidence=self.figure.add_subplot(122)
        self.canvas=FigureCanvasTkAgg(self.figure,master=self);self.canvas.get_tk_widget().pack(fill='both',expand=True)
        self.slider=ttk.Scale(self,from_=0,to=1,command=self.inspect);self.slider.pack(fill='x',padx=15)
        self.details=tk.Text(self,height=7,font=('Consolas',10));self.details.pack(fill='x',padx=10,pady=8)
        self.after(100,self.pump);self.protocol('WM_DELETE_WINDOW',self.close)

    def choose(self,var,kind):
        path=filedialog.askdirectory(initialdir=ROOT) if kind=='directory' else filedialog.askopenfilename(initialdir=ROOT,filetypes=[('JSON','*.json')])
        if path:var.set(path)

    def background(self,fn):
        if self.worker_active:messagebox.showinfo('Working','Wait for the current processing operation.');return
        self.worker_active=True;self.process_button.configure(state='disabled')
        def run():
            try:self.messages.put(('result',fn()))
            except Exception as exc:self.messages.put(('error',str(exc)))
            finally:self.messages.put(('done',None))
        threading.Thread(target=run,daemon=True).start()

    def listen(self):
        if self.listen_button['text']=='Stop receiver':
            self.receiver_stop.set();self.listen_button.configure(text='Start receiver');return
        try:
            port=int(self.port.get())
            if not 1<=port<=65535 or not self.token.get():raise ValueError('Enter port 1–65535 and a token')
        except ValueError as exc:messagebox.showerror('Connection',str(exc));return
        self.receiver_stop=threading.Event();token=self.token.get();stop=self.receiver_stop
        def run():
            try:serve('0.0.0.0',port,token,ROOT/'recordings',lambda text:self.messages.put(('status',text)),stop)
            except Exception as exc:self.messages.put(('error',str(exc)))
        threading.Thread(target=run,daemon=True).start();self.listen_button.configure(text='Stop receiver')

    def import_capture(self):
        path=filedialog.askopenfilename(filetypes=[('Phone recording','*.jsonl')])
        if path:self.background(lambda:('import',str(import_phone(path,ROOT/'recordings'))))

    def process(self):
        args=(self.dataset.get(),self.config.get(),self.prior.get() or None,not self.baseline.get(),self.smooth.get())
        self.status.set('Processing timestamps, visual tracks, IMU and marker constraints…')
        self.background(lambda:('processed',process(*args)))

    def load_results(self):
        path=filedialog.askdirectory(initialdir=ROOT)
        if path:
            try:self.display(Path(path))
            except Exception as exc:messagebox.showerror('Results',str(exc))

    def display(self,output):
        self.output=output
        self.rows=[json.loads(line) for line in (output/'trajectory.jsonl').read_text().splitlines()]
        self.markers=json.loads((output/'marker_map.json').read_text())['markers']
        self.axes.clear();self.confidence.clear()
        positions=np.array([np.array(r['T_world_body'])[:3,3] for r in self.rows])
        self.axes.plot(*positions.T,label='MSCKF',color='#2878a5')
        optimized=json.loads((output/'optimized_trajectory.json').read_text())
        if optimized:
            ps=np.array([np.array(r['T_world_body'])[:3,3] for r in optimized]);self.axes.plot(*ps.T,label='Map refinement',color='#e89428')
        for key,m in self.markers.items():
            p=np.array(m['T_world_marker'])[:3,3];self.axes.scatter(*p,marker='s',color='black');self.axes.text(*p,' '+key)
        self.axes.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title='Trajectory and ArUco map');self.axes.legend()
        ts=np.array([r['timestamp_s'] for r in self.rows]);ts-=ts[0]
        sig=np.array([r['position_sigma_m'] for r in self.rows])
        for i,label in enumerate('XYZ'):self.confidence.plot(ts,2*sig[:,i],label=label)
        self.confidence.set(xlabel='Time (s)',ylabel='Approximate 2σ position uncertainty (m)',title='Estimator uncertainty');self.confidence.legend();self.confidence.grid(alpha=.2)
        self.figure.tight_layout();self.canvas.draw();self.slider.configure(to=max(1,len(self.rows)-1));self.inspect(0)
        self.status.set('Results: '+str(output))

    def inspect(self,value):
        if not self.rows:return
        r=self.rows[min(int(float(value)),len(self.rows)-1)]
        self.details.delete('1.0','end');self.details.insert('end',json.dumps({k:r.get(k) for k in ('timestamp_s','tracking','position_sigma_m','sensors','active_constraints','visible_markers','last_update')},indent=2))

    def evaluate(self):
        if not hasattr(self,'output'):messagebox.showinfo('Evaluation','Load or process a result first.');return
        path=filedialog.askopenfilename(filetypes=[('Ground truth CSV','*.csv')])
        if path:self.background(lambda:('evaluation',evaluate(self.output/'trajectory.csv',path)))

    def pump(self):
        try:
            while True:
                kind,value=self.messages.get_nowait()
                if kind=='status':self.status.set(value)
                elif kind=='error':self.status.set(value);messagebox.showerror('SLAM',value)
                elif kind=='done':self.worker_active=False;self.process_button.configure(state='normal')
                elif kind=='result':
                    label,data=value
                    if label=='import':self.dataset.set(data);self.status.set('Imported '+data)
                    elif label=='processed':self.display(data[0])
                    elif label=='evaluation':
                        (self.output/'evaluation.json').write_text(json.dumps(data,indent=2));messagebox.showinfo('Evaluation',json.dumps(data,indent=2))
        except queue.Empty:pass
        self.after(100,self.pump)

    def close(self):self.receiver_stop.set();self.destroy()


if __name__=='__main__':Dashboard().mainloop()
