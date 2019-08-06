import serial
import pygame.midi
import Adafruit_ADS1x15
import multiprocessing
import Adafruit_GPIO.SPI as SPI
import Adafruit_SSD1306
import numpy as np
from RPi import GPIO
from time import sleep
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from multiprocessing import Queue
from model_functions import create_model, sample_model, sample


def take_sample(ns, sample_taken_event, sample_playing_event, sample_length, model_path, disp, screen_text, change_model_event, q):

    init_display_text = False

    # Hyperparameters:
    VOCABULARY_SIZE = 130 # known 0-127 notes + 128 note_off + 129 no_event
    HIDDEN_UNITS = 256

    # Create model
    model_dec = create_model(VOCABULARY_SIZE, HIDDEN_UNITS, model_path)
    model_dec.reset_states() # Start with LSTM state blank
    seed = 60
    s = sample_model(seed, model_dec, length=32, temperature=1.0)

    # Set up ADS1115
    MAX_ADC_VALUE = 26345
    GAIN = 1
    adc = Adafruit_ADS1x15.ADS1115()

    # Begin sampling loop
    while True:

        # Wait for sample to start playing
        sample_playing_event.wait()

        # Do this to show active models on display after startup
        if not init_display_text:
            oled_print(screen_text, disp)
            init_display_text = True

        # Load new model if it has been changed
        if change_model_event.is_set():
            change_model_event.clear()
            name = q.get()
            model_dec.load_weights(name)
            model_dec.reset_states() # Start with LSTM state blank
            s = sample_model(60, model_dec, length=32, temperature=1.0) # start with dummy sample because of blank states

        try:
            # Read temperature value from adc and convert to number between 0 and 10
            adc_raw = adc_raw = adc.read_adc(0, gain=GAIN)
            temperature = abs(adc_raw) / MAX_ADC_VALUE * 10 + 0.05
            tempo_raw = adc.read_adc(1, gain=GAIN)
            tempo_ms = 1-round(np.log(1+tempo_raw) / np.log(MAX_ADC_VALUE), 3) + 0.125
            #bpm = round(60/(4*tempo_ms))  # b[bpm] = 60[s] / (4 * t[ms])
        except:
            temperature = 1.0
            tempo = 0.183

        # Take sample
        sample = sample_model(seed, model_dec, length=sample_length-1, temperature=temperature)
        ns.value = [sample, tempo_ms]
        seed = sample[-1] # use last sampled note as seed for next prediction

        sample_taken_event.set()
        sample_playing_event.clear()

def play_sample(ns, sample_taken_event, sample_playing_event, player):
    last_played_note = None

    while True:
        # Wait for sample
        sample_taken_event.wait()
        sample_taken_event.clear()
        sample = ns.value[0]
        note_length = ns.value[1]

        sample_playing_event.set()
        play_sequence(player, sample, note_length, last_played_note)

        sample_filtered = sample[sample < 128]
        if len(sample_filtered) > 0:
            last_played_note = sample_filtered[-1]

def play_sequence(player, seq, note_length, first_note=None):
    if first_note:
        last_played_note = first_note
    else:
        last_played_note = seq[0]

    for note in seq:
        next_note = note

        if next_note == 129:
            sleep(note_length)
        elif next_note == 128:
            player.note_off(last_played_note, 127)
            sleep(note_length)
        else:
            player.note_off(last_played_note, 127)
            player.note_on(next_note, 127)
            last_played_note = note
            sleep(note_length)

def rot_instrument_callback(channel):
    global clk_instrument_last_state
    global instrument
    global player
    global instrument_list
    num_instruments = len(instrument_list)

    try:
        clk_state = GPIO.input(CLK_INSTRUMENT)
        if clk_state != clk_instrument_last_state:
            dt_state = GPIO.input(DT_INSTRUMENT)
            if (dt_state != clk_state) and (instrument < num_instruments):
                instrument += 1
            elif instrument > 0:
                instrument -= 1
            player.set_instrument(instrument_list[instrument])
            print(instrument_list[instrument])
            clkLastState = clkState
    except:
        pass

def rot_model_callback(channel):
    global clk_model_last_state
    global model_num
    global models
    global screen_text

    try:
        clk_state = GPIO.input(CLK_MODEL)
        if clk_state != clk_model_last_state:
            dt_state = GPIO.input(DT_MODEL)
            if (dt_state != clk_state) and (model_num < len(models)):
                model_num += 1
            elif model_num > 0:
                model_num -= 1
            screen_text.update_next_model(models[model_num][0])
            oled_print(screen_text, disp)
            clkLastState = clkState
    except:
        pass


def rotary_model_switch_callback(channel):

    global models
    global model_num
    global screen_text
    screen_text.update_model_name(models[model_num][0])
    oled_print(screen_text, disp)
    q.put(models[model_num][1])
    change_model_event.set()

def oled_print(screen_text, disp, loading=False):
    width = disp.width
    height = disp.height
    image = Image.new('1', (width, height))
    padding = 2
    shape_width = 20
    top = padding

    # Get drawing object to draw on image.
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    if not loading:
        draw.text((5, top), "Current model:", font=font, fill=255)
        draw.text((5, top+14), screen_text.model_name, font=font, fill=255)
        draw.text((5, top+32), "Next model:", font=font, fill=255)
        draw.text((5, top+44), screen_text.next_model, font=font, fill=255)
    else:
        draw.text((5, top), "LOADING...", font=font, fill=255)
    disp.image(image)
    disp.display()

class ScreenText():
    def __init__(self):
        self.model_name = ""
        self.next_model = ""

    def update_model_name(self, name):
        #self.model_name = "Model: " + name
        self.model_name = name

    def update_next_model(self, name):
        self.next_model = name



if __name__ == '__main__':

    # There are 128 instruments (0-127), but some are removed because the volume is much lower
    instrument_list = [0,3,4,5,8,11,12,13,14,15,16,20,21,22,23,26,38,40,41,42,43, \
                       44,45,46,47,48,52,53,54,55,60,61,62,64,68,69,70,71,72,73, \
                       74,80,82,83,85,86,87,97,102,104,106,109,110,113,114,115]

    # Rotary encoder for tempo
    CLK_INSTRUMENT = 17
    DT_INSTRUMENT = 18
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(CLK_INSTRUMENT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(DT_INSTRUMENT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(CLK_INSTRUMENT, GPIO.FALLING,
                          callback=rot_instrument_callback,
                          bouncetime=100)
    clk_instrument_last_state = GPIO.input(CLK_INSTRUMENT)


    # Rotary encoder for model change
    CLK_MODEL = 21
    DT_MODEL = 20
    SW_MODEL = 26
    GPIO.setup(CLK_MODEL, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(DT_MODEL, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(SW_MODEL, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    GPIO.add_event_detect(CLK_MODEL, GPIO.FALLING,
                          callback=rot_model_callback,
                          bouncetime=300)
    GPIO.add_event_detect(SW_MODEL, GPIO.RISING,
                          callback=rotary_model_switch_callback,
                          bouncetime=300)
    clk_model_last_state = GPIO.input(CLK_MODEL)

    TIMIDITY_PORT = 3
    instrument = 0
    model_num = 0

    pygame.midi.init()
    player = pygame.midi.Output(TIMIDITY_PORT)
    player.set_instrument(instrument)

    models = [["Bach chorales", "/home/pi/rt_pi/weights/bach.hdf5"],
              ["Final Fantasy 7", "/home/pi/rt_pi/weights/ff7.hdf5"],
              ["Ryan's Mammoth col.", "/home/pi/rt_pi/weights/mammoth.hdf5"]]

    model_path = models[0][1]
    sample_length = 2 # 16 is one bar


    q = Queue()
    RST = 24
    disp = Adafruit_SSD1306.SSD1306_128_64(rst=RST)
    disp.begin()

    screen_text = ScreenText()
    screen_text.update_model_name(models[0][0])
    screen_text.update_next_model(models[0][0])
    oled_print(screen_text, disp, loading=True)


    mgr = multiprocessing.Manager()
    namespace = mgr.Namespace()
    raw_temperature = mgr.Namespace()
    sample_taken_event = multiprocessing.Event()
    sample_playing_event = multiprocessing.Event()
    sample_played_event = multiprocessing.Event()
    sample_playing_event.set()


    change_model_event = multiprocessing.Event()

    ts = multiprocessing.Process(target=take_sample,
                                 args=(namespace,
                                       sample_taken_event,
                                       sample_playing_event,
                                       sample_length,
                                       model_path,
                                       disp,
                                       screen_text,
                                       change_model_event,
                                       q))

    ps = multiprocessing.Process(target=play_sample,
                                 args=(namespace,
                                       sample_taken_event,
                                       sample_playing_event,
                                       player))

    ts.start()
    ps.start()
    ts.join()
    ps.join()


    player.close()
    GPIO.cleanup()
