We need to revamp this app, right now something is off.

1. system requirements:
  - we're using logging and global LOG logger with info/debug levels.
  - we're using Path from pathlib, PySide6, CuPy and whatever else needs, but we're keeping the list of dependencies minimal


2. UI:
  - I need a Button2 class that generates signals for both left and right clicks
  - I need a PySide6 UI, QMainWindow with menu, toolbar and statusbar
  - Toolbar has:
  	* Buttons section:
      - start
      - pause
      - reset
      - step
    * Algo section:
      - Dropbown list with algorithms
    * Draw section:
      - Draw tool brush size, 1-100
      - Draw tool color swatch
    * Clipboard section:
      - 4 Button2 buttons with "clipboards":
        - left click saves
        - right click loads the field.
        - If stored image is of a wrong size - it gets scaled to the current field size on restore
  - All toolbar section are in grid layout with proportional sizes set

  - Menu has:
    - "File-Open" and "Recent files" items.
    - When file is opened, it's added to recent files list(if it's not already in there
    - if it's in there - moved to the top of recents list).
    - Both functions use the same method to load the image and resize it to the current field and set it as a reset image.

  - Hotkeys:
    - Esc: exit
    - Space: play/pause
    - Arrow up - play/pause
    - Arrow left - reset
    - Arrow right - step forward
    - R - reset simulation (set generation counter to 0 and reset the field to the previous stored reset image)
    - H - reset viewport camera position
    - Left mouse click - draw tool
    - Right mouse click - erase tool
    - Middle mouse click - pan
    - Mouse wheel - zoom
    - [ and ] - decrease and increase brush size by 10%


3. Draw tool
  - On left mouse clicks it paints with current color
  - On right mouse it erases the cells
  - Drawing is implemented in a single CUDA kernel with antialiased line

4. On start we're allocating 2 CUDA buffers for 2 fields.
  - It's size should be tied to the widget size pixel to pixel, one to one.
  - On window resize we should keep the old content and just increase or decrease the field size.
  - Image loading and drawing in generation 0 set the "reset image" - next time the app resets, it will reset to this image.
  - Image is resized to the size of the field.
  - We're using PySide6 image handling
  - Kernel goes thru the field and fills in the 2nd field then swaps buffers for display


5. we will rewrite CUDA kernels completely.
  - We will create a single kernel will look around for nearby cells and make a decision based on what it sees.
  -  The decision function will differ depending on algo.
  - We will always use RGBA pixels for all algorithms. Algorithms working with binary logic will process RGBA channels like 4 different games - comparing just R, then just G, then B, then A and such.

6. List of algos:
  - Classic conway game of life
  - Gradual conway game of life: cels are not instantly die and born, but increase and decrease by 10% clamped to 0..255 values.
  - Other types of game of life
