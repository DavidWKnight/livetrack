source ../pysdr/bin/activate

Flight notes:
    The B777 flight to/from taipai gets into ONT around 7pm and leaves around 11pm on Monday, Wednesday, Friday, and Saturday

Monitor output directory size:
    watch "ls -la --block-size=M collects/"
    Might want to pipe to less to fit on one screen

Create tmux script to create the windows:
    dump1090
    htop
    watch "ls -la --block-size=M collects/"
    python livetrack.py

Compression of rf data needs to happen on another thread, else rf thread won't be able to keep up

MVP:
Receive SBS1 messages from dump1090, parse for airborn position messages, when something enters the FOV collect RF data to a file along with ADSB data
No viewer at all besides dump1090.


