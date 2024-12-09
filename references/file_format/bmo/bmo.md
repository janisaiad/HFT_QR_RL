# Appendix IV. Orders Format

Bookmap allows recording and replaying users’ orders with all the events that occur during each order's lifetime. Like the market data, orders files are stored by Bookmap in the C:\Bookmap\Feeds folder in an encrypted format, named as '*bmo', and are automatically replayed with corresponding market data. Traders can also export and import orders files in human-readable format, which implies that such files can be edited or even generated from scratch (e.g., from an external simulation software). This page describes the format of the file.

## Human-Readable Orders files

Orders files can be exported via File >> Export >> Export orders as plain text. The files may contain lines with comments starting with '#'. Each line except the header and comments represents a particular event of a specific order where the type of the event is represented by a single letter as shown below.

## Exported Order Data Example

Here is a simple scenario that includes three orders and shows a corresponding text file. For simplicity, in this example, there is only one working order at a time.

screenshot

!BOOKMAP_FORMAT_V1
!DO_NOT_UPDATE_AFTER_EXECUTION
S,20180817,132604,0.278443826,1105671107,ESU8.CME@RITHMIC,1,2837.0,2
C,20180817,132606,0.660462608,1105671107
S,20180817,132608,0.448465391,1105671108,ESU8.CME@RITHMIC,0,2838.0,2
E,20180817,132609,0.385849739,1105671108,2838.0,2
F,20180817,132609,0.386520348,1105671108
S,20180817,132612,0.325328,1105671115,ESU8.CME@RITHMIC,1,2837.25,2
U,20180817,132614,0.282056116,1105671115,2837.5,2
U,20180817,132617,0.837655884,1105671115,2837.75,2
U,20180817,132621,0.738012058,1105671115,2838.25,2
U,20180817,132625,0.130369159,1105671115,2838.5,2
E,20180817,132625,0.130665043,1105671115,2838.5,2
F,20180817,132625,0.130980406,1105671115

## File Header

The header is needed for internal use to allow backwards compatibility in case the format changes in the future.

!BOOKMAP_FORMAT_V1
!DO_NOT_UPDATE_AFTER_EXECUTION

## Timestamps

Each line has a timestamp represented by three fields: date, hours in UTC timezone, and subseconds. For instance, here it's 17-Aug-2018 13:26:04, 278 milliseconds, 443 microseconds, 826 nanoseconds. The subseconds field can be anything in the range [0,1).

<date>,<time>,<subsecond>
20180817,132604,0.278443826

## Send order message

The first event for any order is always called ‘Send Order’, which starts with S and includes the order's properties and its unique Order ID. This ID will be used for referencing the order during its lifetime:

S,<date>,<time>,<subsecond>,<order id>,<instrument alias>,<bid(1) or ask(0)>,<limit price>,<stop price>,<stop triggered>,<size>
S,20180817,132604,0.278443826,1105671107,ESU8.CME@RITHMIC,1,2837.0,2

## Update order

An update order event starts with U and appears when price and / or size of the order are modified:

U,<date>,<time>,<subsecond>,<order id>,<new limit price>,<new stop price>,<stop triggered>,<new size>
U,20180817,132621,0.738012058,1105671115,2838.25,2

## Cancel order

A cancel order event starts with C:

C,<date>,<time>,<subsecond>,<order id>
C,20180817,132606,0.660462608,1105671107

## Execution

Full or partial execution events start with E and tell at what price and how much of the order was executed:

E,<date>,<time>,<subsecond>,<order id>,<price>,<size>
E,20180817,132625,0.130665043,1105671115,2838.5,2

## Fill

A fill event (F) allows to mark an order as filled (it will be displayed as 0 size by some versions):

F,<date>,<time>,<subsecond>,<order id>
F,20180817,132625,0.130980406,1105671115
