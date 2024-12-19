
from obspy.clients.syngine import Client

# client = Client()

# Get and print available models
# models = client.get_available_models()
# print("Available Models:", models)


from obspy import read
st = read("https://examples.obspy.org/GR.BFO..LHZ.2012.108")
# print(type(st))
# print(st)
# print(st[0].stats)
st.filter("lowpass", freq=0.1, corners=2)
st.plot(type="dayplot", interval=60, right_vertical_labels=False,
        vertical_scaling_range=5e3, one_tick_per_line=True,
        color=['k', 'r', 'b', 'g'], show_y_UTC_label=False,
        events={'min_magnitude': 6.5})
