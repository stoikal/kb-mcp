# Home Server Guide

## Firewall

Allow the service port only from the local subnet. A firewall rule that limits inbound traffic keeps guests on the Wi-Fi from reaching the server.

## Backups

Run pg_dump on a schedule and keep the last two weeks of dumps. Test a restore occasionally so the backups are known to work.

## Fixed address

Reserve the machine's address in the router with a DHCP reservation so client URLs keep working after a reboot.
