from .maxrects import MaxRectsBssf

import operator
import itertools
import collections
import random

import decimal

# Float to Decimal helper
def float2dec(ft, decimal_digits):
    """
    Convert float (or int) to Decimal (rounding up) with the
    requested number of decimal digits.

    Arguments:
        ft (float, int): Number to convert
        decimal (int): Number of digits after decimal point

    Return:
        Decimal: Number converted to decima
    """
    with decimal.localcontext() as ctx:
        ctx.rounding = decimal.ROUND_UP
        places = decimal.Decimal(10)**(-decimal_digits)
        return decimal.Decimal.from_float(float(ft)).quantize(places)


# Sorting algos for rectangle lists
SORT_AREA  = lambda rectlist: sorted(rectlist, reverse=True, 
        key=lambda r: r[0]*r[1]) # Sort by area

SORT_PERI  = lambda rectlist: sorted(rectlist, reverse=True, 
        key=lambda r: r[0]+r[1]) # Sort by perimeter

SORT_DIFF  = lambda rectlist: sorted(rectlist, reverse=True, 
        key=lambda r: abs(r[0]-r[1])) # Sort by Diff

SORT_SSIDE = lambda rectlist: sorted(rectlist, reverse=True, 
        key=lambda r: (min(r[0], r[1]), max(r[0], r[1]))) # Sort by short side

SORT_LSIDE = lambda rectlist: sorted(rectlist, reverse=True, 
        key=lambda r: (max(r[0], r[1]), min(r[0], r[1]))) # Sort by long side

SORT_RATIO = lambda rectlist: sorted(rectlist, reverse=True,
        key=lambda r: r[0]/r[1]) # Sort by side ratio

SORT_NONE = lambda rectlist: list(rectlist) # Unsorted

BoxOrientation = collections.namedtuple("BoxOrientation",
        ["width", "height", "face", "rotated", "vertical"])

MAX_SURFACE_EDGE = 50



class BinFactory(object):

    def __init__(self, width, height, count, pack_algo, *args, **kwargs):
        self._width = width
        self._height = height
        self._count = count
        
        self._pack_algo = pack_algo
        self._algo_kwargs = kwargs
        self._algo_args = args
        self._ref_bin = None # Reference bin used to calculate fitness
        
        self._bid = kwargs.get("bid", None)

    def _create_bin(self):
        return self._pack_algo(self._width, self._height, *self._algo_args, **self._algo_kwargs)

    def is_empty(self):
        return self._count<1

    def fitness(self, width, height):
        if not self._ref_bin:
            self._ref_bin = self._create_bin()

        return self._ref_bin.fitness(width, height)

    def fits_inside(self, width, height):
        # Determine if rectangle widthxheight will fit into empty bin
        if not self._ref_bin:
            self._ref_bin = self._create_bin()

        return self._ref_bin._fits_surface(width, height)

    def new_bin(self):
        if self._count > 0:
            self._count -= 1
            return self._create_bin()
        else:
            return None

    def __eq__(self, other):
        return self._width*self._height == other._width*other._height

    def __lt__(self, other):
        return self._width*self._height < other._width*other._height

    def __str__(self):
        return "Bin: {} {} {}".format(self._width, self._height, self._count)



class PackerBNFMixin(object):
    """
    BNF (Bin Next Fit): Only one open bin at a time.  If the rectangle
    doesn't fit, close the current bin and go to the next.
    """

    def add_box(self, width, height, depth, rid=None):
        orientations = tuple(self._box_orientations(width, height, depth))
        if not orientations:
            return None

        while True:
            if len(self._open_bins) == 0:
                new_bin = self._new_open_bin(orientations=orientations, rid=rid)
                if new_bin is None:
                    return None

            target_bin = self._open_bins[0]
            orientation = self._best_orientation_for_bin(target_bin, orientations)
            if orientation:
                rect = target_bin.add_rect(orientation.width, orientation.height,
                        rid=self._wrap_rid(rid, orientation))
                if rect is not None:
                    return rect

            closed_bin = self._open_bins.popleft()
            self._closed_bins.append(closed_bin)

    def add_box_random(self, width, height, depth, rid=None):
        orientations = tuple(self._box_orientations(width, height, depth))
        if not orientations:
            return None

        while True:
            if len(self._open_bins) == 0:
                new_bin = self._new_open_bin(orientations=orientations, rid=rid)
                if new_bin is None:
                    return None

            target_bin = self._open_bins[0]
            orientation = self._random_orientation_for_bin(target_bin, orientations)
            if orientation:
                rect = target_bin.add_rect(orientation.width, orientation.height,
                        rid=self._wrap_rid(rid, orientation))
                if rect is not None:
                    return rect

            closed_bin = self._open_bins.popleft()
            self._closed_bins.append(closed_bin)

    def add_rect(self, width, height, rid=None):
        while True:
            # if there are no open bins, try to open a new one
            if len(self._open_bins)==0:
                # can we find an unopened bin that will hold this rect?
                new_bin = self._new_open_bin(width, height, rid=rid)
                if new_bin is None:
                    return None

            # we have at least one open bin, so check if it can hold this rect
            rect = self._open_bins[0].add_rect(width, height, rid=rid)
            if rect is not None:
                return rect

            # since the rect doesn't fit, close this bin and try again
            closed_bin = self._open_bins.popleft()
            self._closed_bins.append(closed_bin)


class PackerBFFMixin(object):
    """
    BFF (Bin First Fit): Pack rectangle in first bin it fits
    """
 
    def add_box(self, width, height, depth, rid=None):
        orientations = tuple(self._box_orientations(width, height, depth))
        if not orientations:
            return None

        for b in self._open_bins:
            orientation = self._best_orientation_for_bin(b, orientations)
            if not orientation:
                continue
            rect = b.add_rect(orientation.width, orientation.height,
                    rid=self._wrap_rid(rid, orientation))
            if rect is not None:
                return rect

        while True:
            new_bin = self._new_open_bin(orientations=orientations, rid=rid)
            if new_bin is None:
                return None

            orientation = self._best_orientation_for_bin(new_bin, orientations)
            if not orientation:
                continue

            rect = new_bin.add_rect(orientation.width, orientation.height,
                    rid=self._wrap_rid(rid, orientation))
            if rect is not None:
                return rect

    def add_rect(self, width, height, rid=None):
        # see if this rect will fit in any of the open bins
        for b in self._open_bins:
            rect = b.add_rect(width, height, rid=rid)
            if rect is not None:
                return rect

        while True:
            # can we find an unopened bin that will hold this rect?
            new_bin = self._new_open_bin(width, height, rid=rid)
            if new_bin is None:
                return None

            # _new_open_bin may return a bin that's too small,
            # so we have to double-check
            rect = new_bin.add_rect(width, height, rid=rid)
            if rect is not None:
                return rect

    def add_box_random(self, width, height, depth, rid=None):
        orientations = tuple(self._box_orientations(width, height, depth))
        if not orientations:
            return None

        for b in self._open_bins:
            orientation = self._random_orientation_for_bin(b, orientations)
            if not orientation:
                continue
            rect = b.add_rect(orientation.width, orientation.height,
                    rid=self._wrap_rid(rid, orientation))
            if rect is not None:
                return rect

        while True:
            new_bin = self._new_open_bin(orientations=orientations, rid=rid)
            if new_bin is None:
                return None

            orientation = self._random_orientation_for_bin(new_bin, orientations)
            if not orientation:
                continue

            rect = new_bin.add_rect(orientation.width, orientation.height,
                    rid=self._wrap_rid(rid, orientation))
            if rect is not None:
                return rect


class PackerBBFMixin(object):
    """
    BBF (Bin Best Fit): Pack rectangle in bin that gives best fitness
    """

    # only create this getter once
    first_item = operator.itemgetter(0)

    def add_box(self, width, height, depth, rid=None):
        orientations = tuple(self._box_orientations(width, height, depth))
        if not orientations:
            return False

        best_choice = None
        for b in self._open_bins:
            orientation = self._best_orientation_for_bin(b, orientations)
            if not orientation:
                continue
            fitness = b.fitness(orientation.width, orientation.height)
            if fitness is None:
                continue
            if best_choice is None or fitness < best_choice[0]:
                best_choice = (fitness, b, orientation)

        if best_choice is not None:
            _, best_bin, orientation = best_choice
            if best_bin.add_rect(orientation.width, orientation.height,
                    rid=self._wrap_rid(rid, orientation)):
                return True

        while True:
            new_bin = self._new_open_bin(orientations=orientations, rid=rid)
            if new_bin is None:
                return False

            orientation = self._best_orientation_for_bin(new_bin, orientations)
            if not orientation:
                continue

            if new_bin.add_rect(orientation.width, orientation.height,
                    rid=self._wrap_rid(rid, orientation)):
                return True

    def add_rect(self, width, height, rid=None):
 
        # Try packing into open bins
        fit = ((b.fitness(width, height),  b) for b in self._open_bins)
        fit = (b for b in fit if b[0] is not None)
        try:
            _, best_bin = min(fit, key=self.first_item)
            best_bin.add_rect(width, height, rid)
            return True
        except ValueError:
            pass    

        # Try packing into one of the empty bins
        while True:
            # can we find an unopened bin that will hold this rect?
            new_bin = self._new_open_bin(width, height, rid=rid)
            if new_bin is None:
                return False

            # _new_open_bin may return a bin that's too small,
            # so we have to double-check
            if new_bin.add_rect(width, height, rid):
                return True

    def add_box_random(self, width, height, depth, rid=None):
        orientations = tuple(self._box_orientations(width, height, depth))
        if not orientations:
            return False

        feasible = []
        for b in self._open_bins:
            orientation = self._random_orientation_for_bin(b, orientations)
            if not orientation:
                continue
            feasible.append((b, orientation))

        random.shuffle(feasible)
        for target_bin, orientation in feasible:
            if target_bin.add_rect(orientation.width, orientation.height,
                    rid=self._wrap_rid(rid, orientation)):
                return True

        while True:
            new_bin = self._new_open_bin(orientations=orientations, rid=rid)
            if new_bin is None:
                return False

            orientation = self._random_orientation_for_bin(new_bin, orientations)
            if not orientation:
                continue

            if new_bin.add_rect(orientation.width, orientation.height,
                    rid=self._wrap_rid(rid, orientation)):
                return True



class PackerOnline(object):
    """
    Rectangles are packed as soon are they are added
    """

    def __init__(self, pack_algo=MaxRectsBssf, rotation=True):
        """
        Arguments:
            pack_algo (PackingAlgorithm): What packing algo to use
            rotation (bool): Enable/Disable rectangle rotation
        """
        self._rotation = rotation
        self._pack_algo = pack_algo
        self.reset()

    def __iter__(self):
        return itertools.chain(self._closed_bins, self._open_bins)

    def __len__(self):
        return len(self._closed_bins)+len(self._open_bins)
    
    def _box_orientations(self, width, height, depth):
        """
        Yield all valid orientation options for a 3D box.
        """
        assert(width > 0 and height > 0 and depth > 0)
        if min(width, height, depth) >= MAX_SURFACE_EDGE:
            raise ValueError(
                "Each box must have at least one edge < {}".format(MAX_SURFACE_EDGE)
            )

        faces = (
            ("xy", width, height, depth),
            ("xz", width, depth, height),
            ("yz", height, depth, width),
        )

        seen = set()
        for face, w, h, vertical in faces:
            for rotated in (False, True):
                if rotated and not self._rotation:
                    continue
                ow, oh = (h, w) if rotated else (w, h)
                signature = (ow, oh, face, rotated)
                if signature in seen:
                    continue
                seen.add(signature)
                if min(ow, oh) < MAX_SURFACE_EDGE:
                    yield BoxOrientation(width=ow, height=oh, face=face,
                            rotated=rotated, vertical=vertical)

    def _best_orientation_for_bin(self, pbin, orientations):
        """
        Return the lowest-waste orientation for a given bin.
        """
        best = None
        for orient in orientations:
            fitness = pbin.fitness(orient.width, orient.height)
            if fitness is None:
                continue
            if best is None or fitness < best[0]:
                best = (fitness, orient)

        if best:
            return best[1]
        return None

    def _random_orientation_for_bin(self, pbin, orientations):
        """
        Return a random feasible orientation for the given bin.
        """
        feasible = []
        for orient in orientations:
            if pbin.fitness(orient.width, orient.height) is not None:
                feasible.append(orient)
        if not feasible:
            return None
        return random.choice(feasible)

    def _wrap_rid(self, rid, orientation):
        """
        Preserve original rid while attaching orientation metadata.
        """
        return {
            "rid": rid,
            "face": orientation.face,
            "rotated": orientation.rotated,
            "vertical": orientation.vertical,
            "width": orientation.width,
            "height": orientation.height,
        }

    def __getitem__(self, key):
        """
        Return bin in selected position. (excluding empty bins)
        """
        if not isinstance(key, int):
            raise TypeError("Indices must be integers")

        size = len(self)  # avoid recalulations

        if key < 0:
            key += size

        if not 0 <= key < size:
            raise IndexError("Index out of range")
        
        if key < len(self._closed_bins):
            return self._closed_bins[key]
        else:
            return self._open_bins[key-len(self._closed_bins)]

    def _new_open_bin(self, width=None, height=None, rid=None, orientations=None):
        """
        Extract the next empty bin and append it to open bins

        Returns:
            PackingAlgorithm: Initialized empty packing bin.
            None: No bin big enough for the rectangle was found
        """
        factories_to_delete = set() #
        new_bin = None

        for key, binfac in self._empty_bins.items():

            # Only return the new bin if the rect fits.
            # (If width or height is None, caller doesn't know the size.)
            if orientations is not None:
                if not any(binfac.fits_inside(o.width, o.height)
                        for o in orientations):
                    continue
            else:
                if not binfac.fits_inside(width, height):
                    continue
           
            # Create bin and add to open_bins
            new_bin = binfac.new_bin()
            if new_bin is None:
                continue
            self._open_bins.append(new_bin)

            # If the factory was depleted mark for deletion
            if binfac.is_empty():
                factories_to_delete.add(key)
       
            break

        # Delete marked factories
        for f in factories_to_delete:
            del self._empty_bins[f]

        return new_bin 

    def add_bin(self, width, height, count=1, **kwargs):
        # accept the same parameters as PackingAlgorithm objects
        kwargs['rot'] = self._rotation
        bin_factory = BinFactory(width, height, count, self._pack_algo, **kwargs)
        self._empty_bins[next(self._bin_count)] = bin_factory

    def rect_list(self):
        rectangles = []
        bin_count = 0

        for abin in self:
            for rect in abin:
                rectangles.append((bin_count, rect.x, rect.y, rect.width, rect.height, rect.rid))
            bin_count += 1

        return rectangles

    def bin_list(self):
        """
        Return a list of the dimmensions of the bins in use, that is closed
        or open containing at least one rectangle
        """
        return [(b.width, b.height) for b in self]

    def validate_packing(self):
        for b in self:
            b.validate_packing()

    def reset(self): 
        # Bins fully packed and closed.
        self._closed_bins = collections.deque()

        # Bins ready to pack rectangles
        self._open_bins = collections.deque()

        # User provided bins not in current use
        self._empty_bins = collections.OrderedDict() # O(1) deletion of arbitrary elem
        self._bin_count = itertools.count()


class Packer(PackerOnline):
    """
    Rectangles aren't packed untils pack() is called
    """

    def __init__(self, pack_algo=MaxRectsBssf, sort_algo=SORT_NONE, 
            rotation=True):
        """
        """
        super(Packer, self).__init__(pack_algo=pack_algo, rotation=rotation)
        
        self._sort_algo = sort_algo

        # User provided bins and Rectangles
        self._avail_bins = collections.deque()
        self._avail_rect = collections.deque()
        self._avail_boxes = collections.deque()

        # Aux vars used during packing
        self._sorted_rect = []
        self._sorted_boxes = []

    def add_bin(self, width, height, count=1, **kwargs):
        self._avail_bins.append((width, height, count, kwargs))

    def add_rect(self, width, height, rid=None):
        self._avail_rect.append((width, height, rid))

    def add_box(self, width, height, depth, rid=None, random_orientation=False):
        self._avail_boxes.append((width, height, depth, rid, random_orientation))

    def add_box_random(self, width, height, depth, rid=None):
        self.add_box(width, height, depth, rid=rid, random_orientation=True)

    def _is_everything_ready(self):
        has_items = self._avail_rect or self._avail_boxes
        return has_items and self._avail_bins

    def _box_sort_key(self, box):
        width, height, depth, *_ = box
        faces = (
            width * height,
            width * depth,
            height * depth,
        )
        return max(faces)

    def _select_box_placer(self, random_orientation=False):
        mixin_classes = (PackerBNFMixin, PackerBFFMixin, PackerBBFMixin)
        for cls in type(self).mro():
            if cls in mixin_classes:
                if random_orientation:
                    return cls.add_box_random
                return cls.add_box
        raise AttributeError("Box placement mixin not available for this packer")

    def pack(self):

        self.reset()

        if not self._is_everything_ready():
            # maybe we should throw an error here?
            return

        # Add available bins to packer
        for b in self._avail_bins:
            width, height, count, extra_kwargs = b
            super(Packer, self).add_bin(width, height, count, **extra_kwargs)

        # If enabled sort rectangles
        if self._sort_algo:
            self._sorted_rect = self._sort_algo(self._avail_rect)
        else:
            self._sorted_rect = list(self._avail_rect)

        # Start packing
        for r in self._sorted_rect:
            super(Packer, self).add_rect(*r)

        # Sort boxes in descending order of maximal face area
        self._sorted_boxes = sorted(self._avail_boxes, key=self._box_sort_key,
                reverse=True)

        for box in self._sorted_boxes:
            width, height, depth, rid, random_orientation = box
            placer = self._select_box_placer(random_orientation)
            placer(self, width, height, depth, rid=rid)

 
class PackerBNF(Packer, PackerBNFMixin):
    """
    BNF (Bin Next Fit): Only one open bin, if rectangle doesn't fit
    go to next bin and close current one.
    """
    pass

class PackerBFF(Packer, PackerBFFMixin):
    """
    BFF (Bin First Fit): Pack rectangle in first bin it fits
    """
    pass
    
class PackerBBF(Packer, PackerBBFMixin):
    """
    BBF (Bin Best Fit): Pack rectangle in bin that gives best fitness
    """
    pass 

class PackerOnlineBNF(PackerOnline, PackerBNFMixin):
    """
    BNF Bin Next Fit Online variant
    """
    pass 

class PackerOnlineBFF(PackerOnline, PackerBFFMixin):
    """ 
    BFF Bin First Fit Online variant
    """
    pass

class PackerOnlineBBF(PackerOnline, PackerBBFMixin):
    """ 
    BBF Bin Best Fit Online variant
    """
    pass


class PackerGlobal(Packer, PackerBNFMixin):
    """ 
    GLOBAL: For each bin pack the rectangle with the best fitness.
    """
    first_item = operator.itemgetter(0)
    
    def __init__(self, pack_algo=MaxRectsBssf, rotation=True):
        """
        """
        super(PackerGlobal, self).__init__(pack_algo=pack_algo,
            sort_algo=SORT_NONE, rotation=rotation)

    def _find_best_fit(self, pbin):
        """
        Return best fitness rectangle from rectangles packing _sorted_rect list

        Arguments:
            pbin (PackingAlgorithm): Packing bin

        Returns:
            key of the rectangle with best fitness
        """
        fit = ((pbin.fitness(r[0], r[1]), k) for k, r in self._sorted_rect.items())
        fit = (f for f in fit if f[0] is not None)
        try:
            _, rect = min(fit, key=self.first_item)
            return rect
        except ValueError:
            return None


    def _new_open_bin(self, remaining_rect):
        """
        Extract the next bin where at least one of the rectangles in
        rem

        Arguments:
            remaining_rect (dict): rectangles not placed yet

        Returns:
            PackingAlgorithm: Initialized empty packing bin.
            None: No bin big enough for the rectangle was found
        """
        factories_to_delete = set() #
        new_bin = None

        for key, binfac in self._empty_bins.items():

            # Only return the new bin if at least one of the remaining 
            # rectangles fit inside.
            a_rectangle_fits = False
            for _, rect in remaining_rect.items():
                if binfac.fits_inside(rect[0], rect[1]):
                    a_rectangle_fits = True
                    break

            if not a_rectangle_fits:
                factories_to_delete.add(key)
                continue
           
            # Create bin and add to open_bins
            new_bin = binfac.new_bin()
            if new_bin is None:
                continue
            self._open_bins.append(new_bin)

            # If the factory was depleted mark for deletion
            if binfac.is_empty():
                factories_to_delete.add(key)
       
            break

        # Delete marked factories
        for f in factories_to_delete:
            del self._empty_bins[f]

        return new_bin 

    def pack(self):
       
        self.reset()

        if not self._is_everything_ready():
            return
        
        # Add available bins to packer
        for b in self._avail_bins:
            width, height, count, extra_kwargs = b
            super(Packer, self).add_bin(width, height, count, **extra_kwargs)
    
        # Store rectangles into dict for fast deletion
        self._sorted_rect = collections.OrderedDict(
                enumerate(self._sort_algo(self._avail_rect)))
        
        # For each bin pack the rectangles with lowest fitness until it is filled or
        # the rectangles exhausted, then open the next bin where at least one rectangle 
        # will fit and repeat the process until there aren't more rectangles or bins 
        # available.
        while len(self._sorted_rect) > 0:

            # Find one bin where at least one of the remaining rectangles fit
            pbin = self._new_open_bin(self._sorted_rect)
            if pbin is None:
                break

            # Pack as many rectangles as possible into the open bin
            while True:
              
                # Find 'fittest' rectangle
                best_rect_key = self._find_best_fit(pbin)
                if best_rect_key is None:
                    closed_bin = self._open_bins.popleft()
                    self._closed_bins.append(closed_bin)
                    break # None of the remaining rectangles can be packed in this bin

                best_rect = self._sorted_rect[best_rect_key]
                del self._sorted_rect[best_rect_key]

                PackerBNFMixin.add_rect(self, *best_rect)





# Packer factory
class Enum(tuple): 
    __getattr__ = tuple.index

PackingMode = Enum(["Online", "Offline"])
PackingBin = Enum(["BNF", "BFF", "BBF", "Global"])


def newPacker(mode=PackingMode.Offline, 
         bin_algo=PackingBin.BBF, 
        pack_algo=MaxRectsBssf,
        sort_algo=SORT_AREA, 
        rotation=True):
    """
    Packer factory helper function

    Arguments:
        mode (PackingMode): Packing mode
            Online: Rectangles are packed as soon are they are added
            Offline: Rectangles aren't packed untils pack() is called
        bin_algo (PackingBin): Bin selection heuristic
        pack_algo (PackingAlgorithm): Algorithm used
        rotation (boolean): Enable or disable rectangle rotation. 

    Returns:
        Packer: Initialized packer instance.
    """
    packer_class = None

    # Online Mode
    if mode == PackingMode.Online:
        sort_algo=None
        if bin_algo == PackingBin.BNF:
            packer_class = PackerOnlineBNF
        elif bin_algo == PackingBin.BFF:
            packer_class = PackerOnlineBFF
        elif bin_algo == PackingBin.BBF:
            packer_class = PackerOnlineBBF
        else:
            raise AttributeError("Unsupported bin selection heuristic")

    # Offline Mode
    elif mode == PackingMode.Offline:
        if bin_algo == PackingBin.BNF:
            packer_class = PackerBNF
        elif bin_algo == PackingBin.BFF:
            packer_class = PackerBFF
        elif bin_algo == PackingBin.BBF:
            packer_class = PackerBBF
        elif bin_algo == PackingBin.Global:
            packer_class = PackerGlobal
            sort_algo=None
        else:
            raise AttributeError("Unsupported bin selection heuristic")

    else:
        raise AttributeError("Unknown packing mode.")

    if sort_algo:
        return packer_class(pack_algo=pack_algo, sort_algo=sort_algo, 
            rotation=rotation)
    else:
        return packer_class(pack_algo=pack_algo, rotation=rotation)
