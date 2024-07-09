from blessed import Terminal
import time
import random

class Visualizer:
    def __init__(self, layer_sizes):
        self.term = Terminal()
        self.layer_sizes = layer_sizes

    def initialize(self):
        print(self.term.clear)
        print(self.term.black_on_khaki(self.term.center('Neural Network Training')))
        print('\n' * 2)
        print(self.term.move_y(self.term.height // 2 - 10))
        
        layer_names = ['Input'] + ['Hidden'] * (len(self.layer_sizes) - 2) + ['Output']
        colors = [self.term.olivedrab1, self.term.dodgerblue1, self.term.dodgerblue1, self.term.orangered1]
        for i, (name, size, color) in enumerate(zip(layer_names, self.layer_sizes, colors)):
            y_pos = self.term.height // 2 - 5 + i * 3
            print(self.term.move_xy(self.term.width // 2 - 10, y_pos) + color(f"{name} Layer: {size} nodes"))
        
        print(self.term.move_xy(0, self.term.height - 2) + "Press Ctrl+C to stop training")

    def update(self, epoch, total_epochs, train_loss, val_loss=None):
        progress = int((epoch + 1) / total_epochs * 50)
        
        with self.term.location(0, 3):
            print(self.term.blue(f"Epoch: {epoch + 1}/{total_epochs}"))
            print(self.term.green(f"Train Loss: {train_loss:.6f}"))
            if val_loss is not None:
                print(self.term.yellow(f"Validation Loss: {val_loss:.6f}"))
            print(self.term.red('Progress: [') + self.term.goldenrod(('=' * progress).ljust(50)) + self.term.red(']'))
        
        # Animate network activity
        for i, size in enumerate(self.layer_sizes):
            y_pos = self.term.height // 2 - 5 + i * 3
            x_pos = self.term.width // 2 + 15
            active_node = random.randint(0, size - 1)
            with self.term.location(x_pos, y_pos):
                nodes = '○' * size
                nodes = nodes[:active_node] + '●' + nodes[active_node+1:]
                print(self.term.khaki(nodes))
        
        time.sleep(0.0033)