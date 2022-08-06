'''
Streamlit app to showcase potential for geothermal energy production on NCS
'''
#%% Packages
#from sqlalchemy import func
import streamlit as st
import pickle
import copy

# import plotly.express as px
import plotly.graph_objects as go
# from  plotly.subplots import make_subplots

#%% Description in markdown

# wide_mode = st.checkbox(label='wide mode', value=False)
# if wide_mode:
# st.set_page_config(layout="wide")


st.title('Proof-of-concept for geothermal energy production at Ekofisk-like offshore reservoir')
# st.header('Abstract')
st.markdown('''
This app aims to showcase results of a simple study on geothermal energy 
production on NCS.  
The study is based on back-of-the-envelope calculations on the top of the results from 
mechanistic reservoir models (Eclipse, E500)
''')
st.subheader('General model setup / main features:')
st.markdown('''
 * Ekofisk-like reservoir
 * production-injection pair with ~500 sm3/day production/injection rate
 * symmetry element of a linear drive (1/4, see the image below)
 * 500m x ~250m x 100m = 25 x 19 x 20 blocks
 * narrow fracture from injector to producer
 * heat influx from overburden and underburden  
 * produced water is reinjected at 50 degc
 * E500 model, DEADOIL option, single phase
 * 7 cases with different fracture permeabilities
 ''')

st.image('.//resources//E1_3D_View_TEMP_01_20_May_2023.png', \
    caption='model snapshot with shown gridblock temperature')

#%%----------------------------------------------------------------------------
st.header('Notes on methodology')

st.markdown(
    '''
    Gross heat yield (measured in a power unit) is calculated as $c_{p,w}(T_{in} - T_{out})Q$, where:
    * $c_{p,w}$: water specific heat ([energy unit]/[temperature unit]/[volume unit])
    * $T_{in} - T_{out}$: difference between inlet and outlet temperature of the power plant ([temperature unit]) - retrieved from the model
    * Q:  rate ([volume unit]/[time unit])  
  
    Efficiency of conversion of heat to electrical energy is defined by user.

    Wellbore friction in the injector is introduced by  
    $\Delta P=A-BQ^2$, where:  
    * $dP$: difference between bottomhole (BHP) and wellhead pressure (WHP, bar)  
    * Q: rate for the entire well (i.e. x4) rather than for the symmetry element (m3/h)  
    * A: hydrostatic pressure of the fluid column (bar)  
    * B: quadratic friction parameter (bar*(m3/h)^3)  
    The required (=WHP) pressure is calculated for the producer as follows:   
     $WHP = BHP - \Delta P$  
    (BHP is retrieved from the numerical model)

    Essentially the same formula and parameters are used for the producer:  
    $\Delta P=A+BQ^2$  
    The lacking pressure (which is to be provided by ESP) is calculated as  
    $\Delta P_{add} = \Delta P - (BHP - 1)$  
    (required WHP is set to 1 bar )

     Power required by pump is calcultated by:  
     $W= Q(p_{out} - p_{in})/\eta$  
     where:  
     * $p_{in}/p_{out}$: input/output pressures  
     * $\eta$: pump efficiency (fraction)  
    '''
    )

st.header('Input parameters / results viewer(s)')

#  $p_{in}$: inlet pressure (assumed 1 bar)
vhc_w = 1000*st.number_input(\
    label='water specific heat (J/kg/K)',\
    value=4182.0, min_value=1e+1, max_value=1e+4)

col1_1, col1_2, col1_3 = st.columns(3)
eff_heat2electricity = col1_1.number_input(\
    label='heat-to-electricity conversion efficiency (fraction)',\
    value=0.7, min_value=0.05, max_value=1.0)

eff_pump = col1_2.number_input(\
    label='''pump efficiencies (fraction)''',\
    value=0.7, min_value=0.05, max_value=1.0)

eff_loss_other = col1_3.number_input(\
    label='''other energy losses and costs       
    (fraction of generated electricity)''',\
    value=0.1, min_value=0.00, max_value=1.0)

col2_1, col2_2 = st.columns(2)
a_coeff = col2_1.number_input(\
    label='A coeff. (bar)',\
    value=3000*9.81*1000/1e+5, min_value=1e+2, max_value=4e+3)

b_coeff = col2_2.number_input(\
    label='B coeff. (bar*(m3/h)^3)',\
    value=1.00e-3, min_value=1e-4, max_value=5e-2, step=1e-5, format='%.3e')

#%%-------Model results viewer-----------------------------------------

# uploading the results
@st.cache
def load_model_results(allow_output_mutation=True):
    pp = [
    ".//cases//fperm=250.pkl",
    ".//cases//fperm=10000.pkl",
    ".//cases//fperm=5000.pkl",
    ".//cases//fperm=2500.pkl",
    ".//cases//fperm=1500.pkl",
    ".//cases//fperm=1000.pkl",
    ".//cases//fperm=500.pkl"
    ]
    RR = {}
    for p in pp:
        with open(p, 'rb') as f:
            R=pickle.load(f)
            fperm = int(R[2]['name'][6:])
            R[2]['description'] = f'fracture permeability of {fperm} md'
            RR[fperm] = R
    return RR
RR_ = load_model_results()

RR = copy.deepcopy(RR_) 

# Calculations
T_out = 50 # degC
for cname in RR:
    R = RR[cname][0]
    # gross heat yield (MW)
    R['FGHY'] = vhc_w*R['WLPR_PROD']/86400*(R['WTEMP_PROD'] - T_out)/1e+6
    RR[cname][1]['FGHY'] = 'MW'
    # gross electricity generated (MW)
    R['LOSS1'] = R['FGHY']*(1-eff_heat2electricity)
    R['FGEG'] = R['FGHY'] - R['LOSS1']
    RR[cname][1]['FGEG'] = 'MW'
    R['WTHP_INJ'] = R['WBHP_INJ'] - (a_coeff - b_coeff*(4*R['WWIR_INJ']/24)**2)
    RR[cname][1]['WTHP_INJ'] = 'BARSA'
    # injection power required (MW)
    R['WPOW_INJ'] = R['WWIR_INJ']/86400*(R['WTHP_INJ'] - 1)/eff_pump*1e+5/1e+6
    RR[cname][1]['WPOW_INJ'] = 'MW'
    R['WTHP_PROD'] = 1
    RR[cname][1]['WTHP_PROD'] = 'BARSA'
    # pressure difference required for lifting
    R['WDP_PROD'] = a_coeff + b_coeff*(4*R['WLPR_PROD']/24)**2
    RR[cname][1]['WDP_PROD'] = 'BARSA'
    # lifting power required (MW)
    R['WPOW_PROD'] =  R['WLPR_PROD']/86400*\
        (R['WDP_PROD'] - (R['WBHP_PROD'] - R['WTHP_PROD']))/eff_pump*1e+5/1e+6
    RR[cname][1]['WPOW_PROD'] = 'MW'

    R['LOSS2'] = R['FGEG']*(eff_loss_other)
    # power generated
    R['FPOW'] = R['FGEG'] - R['LOSS2'] - R['WPOW_PROD'] - R['WPOW_INJ'] 
    RR[cname][1]['FPOW'] = 'MW'

dict_mnemonics= \
{'FGHY': 'FGHY: gross heat yield',
 'FGEG': 'FGEG: gross electricity generated',
 'WPOW_INJ': 'WPOW_INJ: injection power cost',
 'WPOW_PROD': 'WPOW_PROD: lifting power cost',
 'FPOW': 'FPOW: net power gain',
 'WBHP_PROD': 'WBHP_PROD: producer BHP',
 'WBHP_INJ': 'WBHP_INJ: injector BHP',
 'WTEMP_PROD': 'WTEMP_PROD: production temperature',
 'WTEMP_INJ': 'WTEMP_INJ: injection temperature',
 'WWIR_INJ': 'WWIR_INJ: injection rate',
 'WWIT_INJ': 'WWIT_INJ: cumulative injection',
 'WLPR_PROD': 'WLPR_PROD: production rate',
 'WWPR_PROD': 'WWPR_PROD: production rate',
 'WLPT_PROD': 'WLPT_PROD: cumulative production',
 'FPR': 'FPR: reservoir pressure',
 'FEPR': 'FEPR: energy production rate',
 'FEPT': 'FEPT: cumulative energy production',
 'FEIR': 'FEIR: energy injection rate',
 'FEIT': 'FEIT: cumulative energy injection',
 'FHLR': 'FHLR: heat loss(+)/influx (-) rate',
 'FHLT': 'FHLT: cumulative heat loss(+)/influx (-)',
 }

mnemonic2description = lambda  x: dict_mnemonics.get(x)
#---------------------------------------------------------
def plotly_mult(RR, y='WTEMP_INJ', axes=None, cases='all', \
    title=None, group_legends_by_case=True, width=None, height=None):
    '''
    creates a Y vs. time chart for specified block_numbers
    RR - one dict, list of dict or dict of dict with IORCoreSim results
    y: str, list
        of mnemonic(s) to plot
    axes : list of int (len(y)=len(axes)) or ndarray
        axes numbers (1,2,3)<=3 to plot vectors Y
    renderer: str
        'jupyterlab', 'browser'
    cases : list
        what cases to plot
    '''
    if isinstance(RR,list): 
        RR = {R[2]['name']:R for R in list(RR)}
    elif isinstance(RR,tuple):
        RR = {RR[2]['name']: RR}  
    elif isinstance(RR,dict): 
        pass
    else:
        raise TypeError(f'RR must be dict or list, rather than {type(RR)}!!!')
        
    if isinstance(y,str): y = [y]
    if len(RR)<=3: clrs=['crimson','forestgreen','royalblue']
    
    for i in RR:  pass
    units = RR[i][1]
    
    # creating a dict of units and corresponding units to display 
    # vectors with same units on the same axis
    axes_vectors = {}
    for i in y:
        if axes_vectors.get(units[i]) == None:
            axes_vectors[units[i]] = [i]
        else:
            axes_vectors[units[i]].append(i) 

    if cases == 'all': cases = RR.keys() 
    many_cases= True if len(cases) >= len(y) else False

    if len(cases)<4:
        clrs = ['crimson','forestgreen','royalblue','darkviolet']*2           
    else:
        clrs = ['crimson','darkorange','gold',
            'forestgreen','deepskyblue',
            'royalblue','darkviolet']*2   

    fig = go.Figure()   
    if y==[]: return fig
    dsh = ['solid', 'dash','dashdot', 'longdash',  'longdashdot', 'dot']
    cc = -1 # case counter
    for name in RR:
        if name in cases:
            cc += 1
            vc = 0
            ac = 0
            for axis_cont,axis_name in zip(axes_vectors.items(),['y','y2','y3','y4']):
                for i, v in enumerate(axis_cont[1]):
                    fig.add_trace(
                        go.Scattergl(x=RR[name][0].index, 
                                     y=RR[name][0][v], 
                                     # name = f'{v} ({name})',
                                     name = v if group_legends_by_case else name,
                                     yaxis = axis_name, 
                                     # hovertext = name, 
                                     legendgroup = name if group_legends_by_case else v,
                                     legendgrouptitle_text = name if group_legends_by_case else v,
                                     line={
                                         'width': 2, 
                                         'color': clrs[cc if many_cases else ac], 
                                         'dash':  dsh[vc]
                                     }))   
                    vc += 1
                ac += 1
    axis_title = []
    for k in axes_vectors:
        mnmncs = str(list(axes_vectors[k])).replace('[','').replace(']','')
        mnmncs =  mnmncs + ' (' + k  + ')'
        axis_title.append(mnmncs) 
        
    if len(axes_vectors) > 1:
        fig.update_layout(
            yaxis2=dict(
                title = axis_title[1],
                titlefont = dict(color=None if many_cases else clrs[1]),
                tickfont=  dict(color=None if many_cases else clrs[1]),
                anchor="free", overlaying="y",
                side="right", position=1.0,
                color = None if many_cases else clrs[1]
            ))
    if len(axes_vectors) > 2:
        fig.update_layout(
            yaxis3=dict(
                title = axis_title[2],
                titlefont = dict(color=None if many_cases else clrs[2]),
                tickfont=  dict(color=None if many_cases else clrs[2]),                
                anchor="free", overlaying="y",
                side="right", position=0.0,
                color = None if many_cases else clrs[2]
            ))
    if len(axes_vectors) > 3:    
        fig.update_layout(
            yaxis4=dict(
                title = axis_title[3],
                titlefont = dict(color=None if many_cases else clrs[3]),
                tickfont=  dict(color=None if many_cases else clrs[3]),                    
                anchor="free", overlaying="y",
                side="left", position=1.0,            
                color = None if many_cases else clrs[3]
            ))    
    
    fig.update_layout(
        title = title,
        template='plotly_white',
        width = width,
        height = height,
        margin=dict(l=10, r=50, t=25, b=25),    
        xaxis = dict(title="time (days)"),
        yaxis = dict(title = axis_title[0], \
            titlefont = dict(color=None if many_cases else clrs[0]),
            tickfont=  dict(color=None if many_cases else clrs[0]),    
        ),
        legend=dict(
            yanchor="top", y=-0.05,
            xanchor="left", x=0, 
            # font={'size': 14}, 
            # traceorder='grouped',
            groupclick="toggleitem",
            orientation = "h",
        ))

    return fig

def waterfall(RR, selected_cases=[1500], selected_day=1500):
    td = selected_day
    fig = go.Figure()   
    for case in selected_cases:
        R = RR[case]
        gross_heat_yield = R[0].loc[td,'FGHY']
        loss_conversion = R[0].loc[td,'LOSS1']
        other_losses = R[0].loc[td,'LOSS2']
        injection_costs = R[0].loc[td,'WPOW_INJ']
        lifting_costs = R[0].loc[td,'WPOW_PROD']

        rest = R[0].loc[td,'FPOW']
        gross_electricity_generated = R[0].loc[td,'FGEG']
        fig = fig.add_trace(
            go.Waterfall(
            name = case, 
            orientation = "v",
            measure=["relative","relative", "total",\
                "relative","relative","relative", "total"],
            x = ['gross heat yield', \
                'loss @ conversion', \
                'electricity generated',\
                'injection costs',  \
                'lifting costs', \
                'other costs', \
                'surplus'],
            textposition = "outside",
            y = [gross_heat_yield,  -loss_conversion, gross_electricity_generated, \
                -injection_costs, -lifting_costs, -other_losses, rest]
        ))

    fig.update_layout(template='plotly_white',\
            waterfallgroupgap = 0.25,
            title = "Energy balance (MW)",
            showlegend = True,
            yaxis=dict(title='MW'),
            height=500
    )
    return fig

#---------------------------------------------------------
col4_1, col4_2 = st.columns(2)
selected_cases = col4_1.multiselect(
    label='select case(s) with fracture permeability, mD:', \
    options=list(RR.keys()), default=[1000, 1500, 2500], key='2',
    )

Y = col4_2.multiselect(
    label='select vectors:', \
    options=list(dict_mnemonics.keys()), \
    default=['FPOW', 'WTEMP_PROD', 'WWIR_INJ'], 
    format_func= mnemonic2description)

fig = plotly_mult(RR, cases=selected_cases, y=Y, height=500)
st.plotly_chart(fig)

selected_day = st.select_slider(label='select day:',\
    value=1500,\
    options = [1, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000])

selected_cases1 = st.multiselect(
        label='select case(s) with fracture permeability, mD:', \
        options=list(RR.keys()), default=[1000, 1500, 2500], key='1',
        )

fig_wtrf = waterfall(RR, selected_cases = selected_cases, selected_day=selected_day)
st.plotly_chart(fig_wtrf)
#%%-- leftovers


# dict_mnemonics = {
# 'WBHP_PROD': 'producer BHP',
# 'WBHP_INJ':  'injector BHP',
# 'WTEMP_PROD': 'production temperature', 
# 'WTEMP_INJ': 'injection temperature', 
# 'WWIR_INJ': 'injection rate', 
# 'WWIT_INJ': 'cumulative injection', 
# 'WWPR_PROD': 'production rate', 
# 'WLPT_PROD': 'cumulative production', 
# 'FPR': 'reservoir pressure',
# 'FEPR': 'energy production rate',
# 'FEPT': 'cumulative energy production',
# 'FEIR': 'energy injection rate',
# 'FEIT': 'cumulative energy injection',
# 'FHLR': 'heat loss(+)/influx (-) rate',
# 'FHLT': 'cumulative heat loss(+)/influx (-)'
# }
